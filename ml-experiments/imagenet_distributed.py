"""
imagenet_distributed.py: Initializes a DenseNet model for distributed multi-label classification 
                         on the MIMIC-CXR dataset, loading and preprocessing data, defining image 
                         paths, and executing distributed training across processes. 

                         Tracks model performance and saves evaluation metrics to a CSV file.

Usage:
python imagenet_distributed.py
"""
import torch
import numpy as np  
import pandas as pd  
import os, os.path
import errno
from tqdm import tqdm
import gc
import encoding_utils
import csv
import argparse
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image
import torchmetrics
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

parser = argparse.ArgumentParser()
parser.add_argument('epochs', type=int,
                    help='Number of Train/Val Epochs')
parser.add_argument('lr', type=float,
                    help='Initial LR')
parser.add_argument('batch_size', type=int,
                    help='Batch Size')

if __name__ == "__main__":
    world_size = 4
    train_df = pd.read_csv('experiments/train.csv.gz', compression='gzip')
    val_df = pd.read_csv('experiments/val.csv.gz', compression='gzip')
    img_prefix = 'mimic-cxr-jpg/2.1.0/files/'
    train_df['images'] = img_prefix + 'p' + (train_df['subject_id'].astype(str)).str[:2] + '/p' + train_df['subject_id'].astype(str) + '/s' + train_df['study_id'].astype(str) + '/' + train_df['dicom_id'].astype(str) + '.jpg'
    val_df['images'] = img_prefix + 'p' + (val_df['subject_id'].astype(str)).str[:2] + '/p' + val_df['subject_id'].astype(str) + '/s' + val_df['study_id'].astype(str) + '/' + val_df['dicom_id'].astype(str) + '.jpg'

    label_columns = ['Atelectasis', 'Cardiomegaly',
           'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
           'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion',
           'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']
    train_df[label_columns] = train_df[label_columns].fillna(0)
    val_df[label_columns] = val_df[label_columns].fillna(0)

    model = torchvision.models.densenet121(weights='DEFAULT')
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(in_features=1024, out_features=len(label_columns)),  # for multi-label output
        torch.nn.Sigmoid()
    )

    batch_size = 128
    max_grad_norm = 0.5
    epochs=64

    mp.spawn(encoding_utils.distributed_training, args=(world_size, model, train_df, val_df, batch_size, max_grad_norm, label_columns, epochs,), nprocs=world_size, join=True)




### Load the best model
##model.load_state_dict(early_stopping.best_model_state)
##        
##    metrics = [{"Training_Accuracy": best_trainacc, 
##               "Name": "DenseNet", 
##               "Validation_Accuracy": best_valacc}]
##    
##    # field names
##    fields = ['Name', 'Training_Accuracy', "Validation_Accuracy"]
##    
##    # name of csv file
##    output_path = "ResNet50/"
##    os.makedirs(os.path.dirname(output_path), exist_ok=True)
##    filename = output_path + str(lr) + "_performance.csv"
##    
##    # writing to csv file
##    print(f"Writing to output file location: {filename}")
##    with open(filename, 'w') as csvfile:
##        # creating a csv dict writer object
##        writer = csv.DictWriter(csvfile, fieldnames=fields)
##        # writing headers (field names)
##        writer.writeheader()
##        # writing data rows
##        writer.writerows(metrics)
##
##    print("Finished!")

