"""
vision_evaluation.py: Processes MIMIC-CXR images to train or evaluate a DenseNet-based
                      vision classifier on multiple medical labels. Loads precomputed
                      images and labels, prepares a dataloader, and applies transformations 
                      for model inference.

Usage:
python vision_evaluation.py [--use_test]
"""

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


# Define image transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Adjust based on your model's expected input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Adjust based on your model's expected normalization
])

def remove_module_prefix(state_dict):
    """
    Remove 'module.' prefix from the keys in the state dictionary.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        # Remove 'module.' prefix
        new_key = k[7:] if k.startswith('module.') else k
        new_state_dict[new_key] = v
    return new_state_dict

# Main function to tie everything together
def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_test', action='store_true', 
                       help='Use test dataset instead of combined train+val dataset')
    args = parser.parse_args()

    # Load datasets
    train_df = pd.read_csv('experiments/train.csv.gz', compression='gzip')
    val_df = pd.read_csv('experiments/val.csv.gz', compression='gzip')
    test_df = pd.read_csv('experiments/experiments.test.csv.gz', compression='gzip')
    img_prefix = 'mimic-cxr-jpg/2.1.0/files/'
    
    # Prepare image paths
    train_df['images'] = img_prefix + 'p' + (train_df['subject_id'].astype(str)).str[:2] + '/p' + train_df['subject_id'].astype(str) + '/s' + train_df['study_id'].astype(str) + '/' + train_df['dicom_id'].astype(str) + '.jpg'
    val_df['images'] = img_prefix + 'p' + (val_df['subject_id'].astype(str)).str[:2] + '/p' + val_df['subject_id'].astype(str) + '/s' + val_df['study_id'].astype(str) + '/' + val_df['dicom_id'].astype(str) + '.jpg'
    test_df['images'] = img_prefix + 'p' + (test_df['subject_id'].astype(str)).str[:2] + '/p' + test_df['subject_id'].astype(str) + '/s' + test_df['study_id'].astype(str) + '/' + test_df['dicom_id'].astype(str) + '.jpg'

    label_columns = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
                      'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']
    train_df[label_columns] = train_df[label_columns].fillna(0)
    val_df[label_columns] = val_df[label_columns].fillna(0)
    test_df[label_columns] = test_df[label_columns].fillna(0)

    # Choose which dataset to use based on flag
    if args.use_test:
        dataset = test_df
        output_dir = 'vision_predictions_test'
    else:
        dataset = pd.concat([train_df, val_df])
        output_dir = 'vision_predictions'

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    dataloader = encoding_utils.vision_dataloaders(dataset, label_columns, transform, batch_size=128, rank=None, world_size=1, shuffle=False)

    # Load model
    model = torchvision.models.densenet121(weights='DEFAULT')
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(in_features=1024, out_features=len(label_columns)),
        torch.nn.Sigmoid()
    )

    state_dict = torch.load("densenet121_vision/best_model.pth")
    fixed_state_dict = remove_module_prefix(state_dict)
    model.load_state_dict(fixed_state_dict)

    # Pass output directory to the prediction function
    encoding_utils.get_vision_predictions(model, dataloader, output_dir=output_dir)
    


# Entry point
if __name__ == "__main__":
    main()
