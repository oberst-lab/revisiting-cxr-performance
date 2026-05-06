import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
import torchmetrics
import tqdm
import encoding_utils
import pickle
import argparse
from sklearn.feature_extraction.text import CountVectorizer


def load_model_outputs(use_test_data=False):
    """
    Load model outputs from appropriate dataset
    """
    file_path = f'vision_predictions_test/full_predictions.pkl' if use_test_data else f'vision_predictions/full_predictions.pkl'
    with open(file_path, 'rb') as file:
        predictions = pickle.load(file)
    return predictions

def chunk(note: str) -> str:
    """Preprocess text by splitting on 'eotextdelimiter' and joining back together."""
    split_note = note.split("eotextdelimiter")
    return " ".join(split_note)


# Main function to tie everything together
def main(label_index, use_test_data=False):
    # Load datasets
    train_df = pd.read_csv('experiments/train.csv.gz', compression='gzip')
    val_df = pd.read_csv('experiments/val.csv.gz', compression='gzip')
    test_df = pd.read_csv('experiments/experiments.test.csv.gz', compression='gzip')

    label_columns = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
                    'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other', 
                    'Pneumonia', 'Pneumothorax', 'Support Devices']
    
    # Handle NA values
    train_df[label_columns] = train_df[label_columns].fillna(0)
    val_df[label_columns] = val_df[label_columns].fillna(0)
    test_df[label_columns] = test_df[label_columns].fillna(0)

    # Select dataset based on flag
    if use_test_data:
        dataset = test_df
        output_parent_dir = 'prior_mention_evaluation_test'
        print("Using TEST dataset for evaluation")
    else:
        dataset = pd.concat([train_df, val_df])
        output_parent_dir = 'prior_mention_evaluation'
        print("Using TRAIN+VAL dataset for evaluation")

    # Load model outputs
    all_outputs = load_model_outputs(use_test_data)
    all_labels = torch.Tensor(dataset[label_columns].values).int()
    label = label_columns[label_index]

    # Vectorize text
    dataset['processed_text'] = dataset['text'].map(chunk)
    
    words_to_analyze = {0: 'atelectasis', 1: 'cardiomegaly',
                        2: 'bronchoscopy', 3: 'lasix',
                        4: 'tachycardic', 5: 'fractures',
                        6: 'metastatic', 7: 'opacities',
                        8: 'lung', 9: 'effusions',
                        10: 'pleural', 11: 'pneumonia',
                        12: 'pneumothorax', 13: 'placement'}
    selected_feature = words_to_analyze[label_index]

    # Analyze mentions
    prior_mention = dataset['processed_text'].str.contains(rf'\b{selected_feature}\b', regex=True)
    positive_indices = np.array(dataset[prior_mention].index)
    #dict_of_mentions = {1: positive_indices}

    negative_indices = np.array(dataset[~prior_mention].index)
    #dict_of_no_mentions = {0: negative_indices}

    (pos_mean, pos_ci, 
         neg_mean, neg_ci, 
         diff_mean, diff_ci) = encoding_utils.bootstrap_auroc_difference_threshold(
            label_index=label_index,
            confidence_level=0,
            confident_indices=positive_indices,
            non_confident_indices=negative_indices,
            num_bootstraps=100000,
            all_outputs=all_outputs,
            all_labels=all_labels
        )

    # Save results
    results = {
        "performance_on_previous_mentions": (pos_mean, pos_ci),
        "performance_on_no_previous_mentions": (neg_mean, neg_ci),
        "diffs": (diff_mean, diff_ci),
        "dataset_used": "test" if use_test_data else "train_val"
    }

    os.makedirs(output_parent_dir, exist_ok=True)
    with open(f'{output_parent_dir}/{label}_evaluation.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results saved to {output_parent_dir}/{label}_evaluation.pkl")


# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate some labels.")
    parser.add_argument('--label_index', type=int, help='Index of the label', required=True)
    parser.add_argument('--use_test_data', action='store_true', help='Use test dataset instead of train+val')
    args = parser.parse_args()
    main(args.label_index, args.use_test_data)
