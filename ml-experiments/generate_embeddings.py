"""
generate_embeddings.py: 
Tokenizes MIMIC notes and performs inference using various HF models.
Handles context window limits and conditionally enables Flash Attention 2.
"""

import argparse
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.multiprocessing as mp
import os
import random
from datetime import timedelta
from transformers import AutoModel, AutoTokenizer, AutoConfig
from datasets import load_dataset, concatenate_datasets

# Map friendly names to HF IDs and Max Lengths
MODEL_CONFIGS = {
    "mistral": ("mistralai/Mistral-7B-Instruct-v0.1", 4096),
    "pubmedbert": ("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", 512),
    "bert": ("bert-base-uncased", 512),
    "clinicalbert": ("emilyalsentzer/Bio_ClinicalBERT", 512),
    "biolinkbert": ("michiyasunaga/BioLinkBERT-base", 512),
    "roberta": ("FacebookAI/roberta-base", 512)
}

class AutoEncoder:
    """Wrapper for HF Models to handle different architectures uniformally"""
    def __init__(self, model_key, device_id=None):
        model_name, max_len = MODEL_CONFIGS[model_key]
        self.max_len = max_len
        self.device = torch.device(f"cuda:{device_id}" if device_id is not None else "cuda")
        
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # --- CONDITIONAL FLASH ATTENTION LOGIC ---
        model_kwargs = {
            "torch_dtype": torch.float16, # FA2 requires float16 or bfloat16
            "trust_remote_code": True
        }

        # BERT/RoBERTa do not support the "flash_attention_2" flag in standard HF implementation.
        # Mistral (and Llama/Falcon) do.
        if "mistral" in model_key:
            try:
                # Check if flash_attn is actually installed
                import flash_attn
                print(f"Enabling Flash Attention 2 for {model_key}...")
                model_kwargs["attn_implementation"] = "flash_attention_2"
            except ImportError:
                print("Warning: flash_attn not installed. Falling back to default attention.")

        self.model = AutoModel.from_pretrained(
            model_name, 
            **model_kwargs
        ).to(self.device)
        self.model.eval()

def mean_pooling(last_hidden, attention_mask):
    """Generic mean pooling for any transformer"""
    expanded_mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    return torch.sum(last_hidden * expanded_mask, 1) / torch.clamp(expanded_mask.sum(1), min=1e-9)

def chunk_text(tokenizer, text, max_len, overlap=50):
    """Chunks text into segments of max_len."""
    tokens = tokenizer(text, add_special_tokens=False, return_tensors='pt')['input_ids'][0]
    effective_len = max_len - 2 
    
    if len(tokens) <= effective_len:
        return [tokens]
    
    chunks = []
    stride = effective_len - overlap
    for i in range(0, len(tokens), stride):
        chunk = tokens[i : i + effective_len]
        chunks.append(chunk)
        if i + effective_len >= len(tokens):
            break
            
    return chunks

class HFWrapper(Dataset):
    def __init__(self, hf_data):
        self.data = hf_data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]['text'], idx

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    job_id = os.environ.get('SLURM_JOB_ID', str(random.randint(10000, 20000)))
    port = 20000 + (int(job_id) % 40000)
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timedelta(hours=12))

def cleanup():
    dist.destroy_process_group()

def process_batch(encoder, batch_notes, batch_indices, chunk_batch_size, device):
    all_embeddings = []
    all_indices = []
    
    for note, idx in zip(batch_notes, batch_indices):
        try:
            input_ids_list = chunk_text(encoder.tokenizer, note, encoder.max_len)
            
            if not input_ids_list:
                all_embeddings.append(torch.zeros(encoder.model.config.hidden_size).to(device))
                all_indices.append(idx)
                continue
            
            note_embeddings = []
            
            for i in range(0, len(input_ids_list), chunk_batch_size):
                current_inputs = input_ids_list[i:i + chunk_batch_size]
                
                # Manual batch construction
                batch_tensors = []
                for inp in current_inputs:
                    batch_tensors.append(inp)

                padded_batch = encoder.tokenizer.pad(
                    {'input_ids': batch_tensors},
                    padding=True,
                    return_tensors="pt"
                ).to(device)
                
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = encoder.model(
                        input_ids=padded_batch['input_ids'], 
                        attention_mask=padded_batch['attention_mask']
                    )
                    
                    if hasattr(outputs, 'last_hidden_state'):
                        last_hidden = outputs.last_hidden_state
                    else:
                        last_hidden = outputs[0]

                    pooled = mean_pooling(last_hidden, padded_batch['attention_mask'])
                    pooled = F.normalize(pooled, p=2, dim=1)
                    note_embeddings.append(pooled)
                
            if note_embeddings:
                note_embedding = torch.mean(torch.cat(note_embeddings, dim=0), dim=0)
            else:
                note_embedding = torch.zeros(encoder.model.config.hidden_size).to(device)
            
            all_embeddings.append(note_embedding)
            all_indices.append(idx)
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"GPU OOM at note {idx}. Saving zero embedding.")
                all_embeddings.append(torch.zeros(encoder.model.config.hidden_size).to(device))
                all_indices.append(idx)
                torch.cuda.empty_cache()
            else:
                print(f"Error processing note {idx}: {e}")
                all_embeddings.append(torch.zeros(encoder.model.config.hidden_size).to(device))
                all_indices.append(idx)
    
    return all_embeddings, all_indices

def run_inference(rank, world_size, args):
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    data_files = {"train": args.train_path}
    if args.val_path:
        data_files["validation"] = args.val_path
    
    full_dataset = load_dataset("csv", data_files=data_files, split="train")
    if args.val_path:
        val_set = load_dataset("csv", data_files=data_files, split="validation")
        full_dataset = concatenate_datasets([full_dataset, val_set])
    
    dataset = HFWrapper(full_dataset)

    encoder = AutoEncoder(args.model_name, device_id=rank)
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0, pin_memory=True)
    
    rank_embeddings = []
    rank_indices = []
    
    with torch.no_grad():
        for batch_idx, (notes, indices) in enumerate(tqdm(dataloader, desc=f"Rank {rank} encoding", position=rank)):
            batch_embeddings, batch_indices = process_batch(encoder, notes, indices, args.chunk_batch_size, device)
            rank_embeddings.extend(batch_embeddings)
            rank_indices.extend(batch_indices)
            if batch_idx > 0 and batch_idx % 20 == 0:
                torch.cuda.empty_cache()
    
    rank_embeddings = [emb.cpu() for emb in rank_embeddings]
    
    temp_path = f"{args.output_path}.temp_rank_{rank}.pt"
    torch.save({'embeddings': rank_embeddings, 'indices': rank_indices}, temp_path)
    
    dist.barrier(device_ids=[rank])
    
    if rank == 0:
        all_embeddings = []
        all_indices = []
        for r in range(world_size):
            temp_file = f"{args.output_path}.temp_rank_{r}.pt"
            if os.path.exists(temp_file):
                data = torch.load(temp_file, map_location='cpu')
                all_embeddings.extend(data['embeddings'])
                all_indices.extend(data['indices'])
                os.remove(temp_file)
        
        result_dict = {idx.item() if isinstance(idx, torch.Tensor) else idx: emb 
                       for idx, emb in zip(all_indices, all_embeddings)}
        
        final_list = []
        for i in range(len(dataset)):
            if i in result_dict:
                final_list.append(result_dict[i])
            else:
                final_list.append(torch.zeros(encoder.model.config.hidden_size))
        
        final_embeddings = torch.stack(final_list)
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(final_embeddings, args.output_path)
        print(f"Saved embeddings to {args.output_path}")
    
    cleanup()

def main(args):
    try:
        for f in Path(args.output_path).parent.glob("*.temp_rank_*.pt"):
            f.unlink()
    except Exception:
        pass

    world_size = torch.cuda.device_count()
    if world_size > 1:
        mp.spawn(run_inference, args=(world_size, args), nprocs=world_size, join=True)
    else:
        print("This script is optimized for DDP. Please run with at least 1 GPU visible.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--val_path', type=str, required=False)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True, choices=MODEL_CONFIGS.keys())
    parser.add_argument('--chunk_batch_size', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()
    main(args)