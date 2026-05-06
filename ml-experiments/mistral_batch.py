"""
mistral.py: Tokenizes MIMIC notes, performs inference using the Mistral
            model to obtain encodings of medical context.
Usage:
python mistral.py --train_path ... --val_path ... --output_path ...
"""

import argparse
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import encoding_utils  # Make sure it contains Mistral_Encoder + chunk()
from pathlib import Path


def mean_pooling(last_hidden, attention_mask):
    expanded_mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    return torch.sum(last_hidden * expanded_mask, 1) / torch.clamp(expanded_mask.sum(1), min=1e-9)


def get_encodings(encoder, df_texts, chunk_batch_size=8):
    encoder.model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = []

    with torch.no_grad():
        for idx, note in enumerate(tqdm(df_texts['text'], desc="Encoding notes")):
            try:
                input_ids_list, attention_masks_list = encoding_utils.chunk(encoder.tokenizer, note)

                # Check if input_ids_list is empty
                if not input_ids_list:
                    print(f"Warning: Empty note at index {idx}, skipping...")
                    # zero vector for empty note
                    zero_embedding = torch.zeros(encoder.model.config.hidden_size)
                    results.append(zero_embedding)
                    continue

                all_embeddings = []

                for i in range(0, len(input_ids_list), chunk_batch_size):
                    batch_input_ids = pad_sequence(
                        input_ids_list[i:i + chunk_batch_size],
                        batch_first=True,
                        padding_value=encoder.tokenizer.pad_token_id
                    ).to(device)

                    batch_attention_mask = pad_sequence(
                        attention_masks_list[i:i + chunk_batch_size],
                        batch_first=True,
                        padding_value=0
                    ).to(device)

                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        outputs = encoder.model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
                        pooled = mean_pooling(outputs.hidden_states[-1], batch_attention_mask)
                        pooled = F.normalize(pooled, p=2, dim=1)
                        all_embeddings.append(pooled)

                    # release memory
                    del batch_input_ids, batch_attention_mask, outputs

                note_embedding = torch.mean(torch.cat(all_embeddings, dim=0), dim=0)
                results.append(note_embedding)

                # clear GPU memory periodically
                if idx > 0 and idx % 20 == 0:
                    torch.cuda.empty_cache()
                    if idx % 100 == 0:
                        current_memory = torch.cuda.memory_allocated() / 1024**2
                        print(f"Processed {idx} notes, GPU memory: {current_memory:.1f}MB")

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"GPU OOM at note {idx}, trying with smaller batch size...")
                    torch.cuda.empty_cache()
                    # smaller batch size
                    single_note_df = pd.DataFrame({'text': [note]})
                    embedding = get_encodings(encoder, single_note_df, chunk_batch_size=max(1, chunk_batch_size//2))
                    results.append(embedding[0])
                else:
                    print(f"Error processing note {idx}: {e}")
                    # zero vector for failed note
                    zero_embedding = torch.zeros(encoder.model.config.hidden_size)
                    results.append(zero_embedding)

    return torch.stack(results).cpu()


def main(args):
    print("Loading data...")
    train_df = pd.read_csv(args.train_path, compression='gzip')

    # Only load val_df if val_path is provided
    if args.val_path:
        val_df = pd.read_csv(args.val_path, compression='gzip')
        full_df = pd.concat([train_df, val_df], ignore_index=True)
    else:
        full_df = train_df

    print("Initializing encoder...")
    encoder = encoding_utils.Mistral_Encoder()
    if encoder.tokenizer.pad_token is None:
        encoder.tokenizer.pad_token = encoder.tokenizer.eos_token

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Using GPU: {gpu_name} ({gpu_memory:.1f}GB)")

    print("Running inference...")
    embeddings = get_encodings(encoder, full_df, chunk_batch_size=args.chunk_batch_size)

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, args.output_path)
    print(f"Saved embeddings to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, required=True, help="Path to train.csv.gz")
    parser.add_argument('--val_path', type=str, required=False, help="Path to val.csv.gz")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save .pt file")
    parser.add_argument('--chunk_batch_size', type=int, default=8, help="Chunk batch size per note")
    args = parser.parse_args()
    main(args)