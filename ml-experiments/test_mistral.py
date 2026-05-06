import os
import torch
import argparse
import pandas as pd
import numpy as np
import csv
import gc
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import encoding_utils

def setup_process(rank, world_size):
    torch.cuda.set_device(rank)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    torch.distributed.destroy_process_group()

def mean_pooling(model_output, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.size()).float()
    return torch.sum(model_output * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_encodings(encoder, df_grouped):
    print("Getting encodings...\n")
    encoder.model.eval()
    preds = []
    with torch.no_grad():
        for note in tqdm(df_grouped['text']):
            chunked_input_ids, chunked_attention_mask = encoding_utils.chunk(encoder.tokenizer, note)
            chunk_embeddings = []
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                for i in range(len(chunked_input_ids)):
                    input_ids = chunked_input_ids[i].unsqueeze(0).cuda()
                    attention_mask = chunked_attention_mask[i].unsqueeze(0).cuda()                 
                    logits = encoder.model(input_ids=input_ids, attention_mask=attention_mask)
                    pooled_output = mean_pooling(logits.hidden_states[-1].cpu(), attention_mask.cpu()).cpu()
                    pooled_output = F.normalize(pooled_output, p=2, dim=1).cpu()              
                    del input_ids, attention_mask, logits
                    torch.cuda.empty_cache()              
                    chunk_embeddings.append(pooled_output)
                avg_embedding = torch.mean(torch.stack(chunk_embeddings), dim=0)
                preds.append(avg_embedding)
        encodings = torch.stack(preds, dim=0)
        encodings = torch.squeeze(encodings).numpy()
    print("Finished passing through model\n")
    return encodings

def main(rank, world_size):
    setup_process(rank, world_size)
    
    # Load the full dataset (all processes load it)
    df_og = pd.read_csv('experiments/experiments.test.csv.gz', compression='gzip')
    
    # Create global indices to track original order
    global_indices = np.arange(len(df_og))
    split_indices = np.array_split(global_indices, world_size)
    
    # Each process gets its portion of the data AND the corresponding indices
    local_df = df_og.iloc[split_indices[rank]]
    local_indices = split_indices[rank]
    
    encoder = encoding_utils.Mistral_Encoder()
    encodings = get_encodings(encoder, local_df)

    # Convert to tensors and prepare for gathering
    encodings_tensor = torch.tensor(encodings, dtype=torch.float32).cuda(rank)
    indices_tensor = torch.tensor(local_indices, dtype=torch.long).cuda(rank)
    
    # Prepare lists for gathering
    gathered_encodings = [torch.zeros_like(encodings_tensor) for _ in range(world_size)]
    gathered_indices = [torch.zeros_like(indices_tensor) for _ in range(world_size)]
    
    # Gather both encodings and their original indices
    torch.distributed.all_gather(gathered_encodings, encodings_tensor)
    torch.distributed.all_gather(gathered_indices, indices_tensor)

    if rank == 0:
        # Combine all gathered tensors
        full_encodings = torch.cat(gathered_encodings).cpu().numpy()
        full_indices = torch.cat(gathered_indices).cpu().numpy()
        
        # Reorder encodings to match original DataFrame order
        ordered_encodings = np.zeros_like(full_encodings)
        ordered_encodings[full_indices] = full_encodings
        
        torch.save(ordered_encodings, 'test.pt')

    cleanup()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Mistral model with Distributed Processing")
    parser.add_argument("--world_size", type=int, default=torch.cuda.device_count(), help="Number of GPUs")
    args = parser.parse_args()
    
    world_size = args.world_size
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)
