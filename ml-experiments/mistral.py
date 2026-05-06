"""
mistral.py: Tokenizes MIMIC notes, performs inference using the Mistral
            model (https://huggingface.co/docs/transformers/main/en/model_doc/mistral) to obtain
            encodings of medical context
Usage:
python mistral.py - writes embeddings out to .pt file
"""

from transformers import AutoTokenizer, AutoModel, pipeline
import datasets
import torch
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from pandas.api.types import CategoricalDtype
from sklearn import metrics
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from typing import List
import os, os.path
import errno
from transformers import DataCollatorWithPadding
from tqdm import tqdm
import gc
import encoding_utils
import csv
import argparse
import torch.nn.functional as F


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(model_output.size()).float()
    )
    return torch.sum(model_output * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def get_encodings(encoder, df_grouped):
    print("Getting encodings...\n")
    encoder.model.eval()
    preds = []
    with torch.no_grad():
        for note in tqdm(df_grouped["text"]):
            chunked_input_ids, chunked_attention_mask = encoding_utils.chunk(
                encoder.tokenizer, note
            )
            chunk_embeddings = []
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                for i in range(len(chunked_input_ids)):
                    input_ids = chunked_input_ids[i].unsqueeze(0).cuda()
                    attention_mask = chunked_attention_mask[i].unsqueeze(0).cuda()

                    logits = encoder.model(
                        input_ids=input_ids, attention_mask=attention_mask
                    )
                    pooled_output = mean_pooling(
                        logits.hidden_states[-1].cpu(), attention_mask.cpu()
                    ).cpu()
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

def main(args):
    train_df = pd.read_csv(args.train_path, compression='gzip')
    val_df = pd.read_csv(args.val_path, compression='gzip')
    df_og = pd.concat([train_df, val_df])
    
    encoder = encoding_utils.Mistral_Encoder()
    encodings = get_encodings(encoder, df_og)
    torch.save(encodings, args.output_path)
    
    # auroc_l1, weight_matrix_l1, accuracy_l1, precision_l1, recall_l1 = encoding_utils.train(labels, encodings, 'l1')
    # auroc_l2, weight_matrix_l2, accuracy_l2, precision_l2, recall_l2 = encoding_utils.train(labels, encodings, 'l2')
    #
    # metrics = [{"AUROC_L1": auroc_l1,
    #            "AUROC_L2": auroc_l2,
    #            "Name": "Mistral",
    #            "Var_L1": np.var(weight_matrix_l1),
    #            "Var_L2": np.var(weight_matrix_l2),
    #            "Model_Output" : "last_hidden_state"}]
    #
    # # field names
    # fields = ['Name', 'AUROC_L1', 'Var_L1', 'AUROC_L2', "Var_L2", "Model_Output"]
    #
    # # name of csv file
    # output_path = "Mistral/"
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # filename = output_path + "performance.csv"
    #
    # # writing to csv file
    # print(f"Writing to output file location: {filename}")
    # with open(filename, 'w') as csvfile:
    #     # creating a csv dict writer object
    #     writer = csv.DictWriter(csvfile, fieldnames=fields)
    #     # writing headers (field names)
    #     writer.writeheader()
    #     # writing data rows
    #     writer.writerows(metrics)

    print("Finished!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_num', type=int, default=0, help="GPU device number to use")
    parser.add_argument('--train_path', type=str, required=True, help="Path to train.csv.gz")
    parser.add_argument('--val_path', type=str, required=True, help="Path to val.csv.gz")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save .pt output")
    args = parser.parse_args()
    main(args)

