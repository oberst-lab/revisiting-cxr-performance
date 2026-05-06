"""
vision_finetuning_foundation.py: Trains a linear classification head on top of frozen
                        ImageNet-pretrained backbone (DenseNet-121, ResNet-50)
                        for MIMIC-CXR multi-label classification, followed by
                        test set evaluation.
"""

import torch
import torch.nn as nn
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch.optim as optim
import os
import pandas as pd
from torchvision import transforms
import torchvision
import tqdm
import encoding_utils
import argparse
import torchmetrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['densenet121', 'resnet50'],
                        help='Which model architecture to train')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the linear head')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--out_dir', type=str, default='./checkpoints', help='Directory to save weights')
    parser.add_argument('--train_csv', type=str, required=True, help='Path to train CSV file')
    parser.add_argument('--val_csv', type=str, required=True, help='Path to val CSV file')
    parser.add_argument('--test_csv', type=str, required=True, help='Path to test CSV file')
    parser.add_argument('--img_prefix', type=str, default='', help='Prefix path to image directory')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of DataLoader worker processes')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Both models use standard ImageNet normalization
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if args.model_type == 'densenet121':
        save_dir = os.path.join(args.out_dir, "densenet121_vision")
    elif args.model_type == 'resnet50':
        save_dir = os.path.join(args.out_dir, "resnet50_vision")

    os.makedirs(save_dir, exist_ok=True)

    print(f"===== Model: {args.model_type} =====")
    print(f"  Epochs:       {args.epochs}")
    print(f"  LR:           {args.lr}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  Num workers:  {args.num_workers}")
    print(f"  Results dir:  {save_dir}")
    print("Loading datasets...")
    train_df = pd.read_csv(args.train_csv, compression='gzip')
    val_df = pd.read_csv(args.val_csv, compression='gzip')
    test_df = pd.read_csv(args.test_csv, compression='gzip')

    for df in [train_df, val_df, test_df]:
        df['images'] = args.img_prefix + 'p' + (df['subject_id'].astype(str)).str[:2] + '/p' + df['subject_id'].astype(str) + '/s' + df['study_id'].astype(str) + '/' + df['dicom_id'].astype(str) + '.jpg'

    label_columns = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
                      'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']
    
    train_df[label_columns] = train_df[label_columns].fillna(0)
    val_df[label_columns] = val_df[label_columns].fillna(0)
    test_df[label_columns] = test_df[label_columns].fillna(0)

    train_loader = encoding_utils.vision_dataloaders(train_df, label_columns, transform, batch_size=args.batch_size, rank=None, world_size=1, shuffle=True, num_workers=args.num_workers)
    val_loader = encoding_utils.vision_dataloaders(val_df, label_columns, transform, batch_size=args.batch_size, rank=None, world_size=1, shuffle=False, num_workers=args.num_workers)
    test_loader = encoding_utils.vision_dataloaders(test_df, label_columns, transform, batch_size=args.batch_size, rank=None, world_size=1, shuffle=False, num_workers=args.num_workers)

    if args.model_type == 'densenet121':
        model = torchvision.models.densenet121(weights='DEFAULT')
        model.classifier = nn.Linear(in_features=1024, out_features=len(label_columns))
        for param in model.features.parameters():
            param.requires_grad = False

    elif args.model_type == 'resnet50':
        model = torchvision.models.resnet50(weights='DEFAULT')
        model.fc = nn.Linear(in_features=2048, out_features=len(label_columns))
        for param in list(model.parameters())[:-2]:  # freeze all except final fc
            param.requires_grad = False

    model = model.to(device)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    
    auroc_metric = torchmetrics.classification.MultilabelAUROC(num_labels=14, average='macro').to(device)
    best_val_auroc = 0.0

    for epoch in range(args.epochs):
        model.train()
        # Keep frozen layers in eval mode to disable dropout/batchnorm updates
        for module in model.modules():
            if not any(p.requires_grad for p in module.parameters(recurse=False)):
                module.eval()
        train_loss = 0.0
        
        train_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device).float()
            
            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(images)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_bar.set_postfix({'loss': loss.item()})
            
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        auroc_metric.reset()
        val_loss = 0.0
        
        with torch.inference_mode():
            val_bar = tqdm.tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device).float()
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
                auroc_metric.update(outputs.float().sigmoid(), labels.long())
                
        avg_val_loss = val_loss / len(val_loader)
        val_auroc = auroc_metric.compute().item()
        
        print(f"Epoch {epoch+1} Summary: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Macro AUROC: {val_auroc:.4f}")

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            save_path = os.path.join(save_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f" Saved new best model to {save_path} (AUROC: {val_auroc:.4f})")
    
    # Load the best weights we just saved
    best_model_path = os.path.join(save_dir, "best_model.pth")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Set up metrics to capture both the macro average and the individual class scores
    test_auroc_macro = torchmetrics.classification.MultilabelAUROC(num_labels=14, average='macro').to(device)
    test_auroc_per_class = torchmetrics.classification.MultilabelAUROC(num_labels=14, average='none').to(device)

    with torch.inference_mode():
        test_bar = tqdm.tqdm(test_loader, desc="Testing")
        for images, labels in test_bar:
            images, labels = images.to(device), labels.to(device).long()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(images)
            test_auroc_macro.update(outputs.float().sigmoid(), labels)
            test_auroc_per_class.update(outputs.float().sigmoid(), labels)

    final_macro_auroc = test_auroc_macro.compute().item()
    final_per_class_auroc = test_auroc_per_class.compute().cpu().numpy()

    print(f" Final Test Results ({args.model_type}):")
    print(f"Macro AUROC: {final_macro_auroc:.4f}")
    
    test_results = []
    for idx, label in enumerate(label_columns):
        score = final_per_class_auroc[idx]
        print(f"  {label}: {score:.4f}")
        test_results.append({'Label': label, 'Test_AUROC': score})
    
    test_results.append({'Label': 'MACRO_AVERAGE', 'Test_AUROC': final_macro_auroc})
    pd.DataFrame(test_results).to_csv(os.path.join(save_dir, 'test_metrics.csv'), index=False)
    print(f" Metrics saved to {os.path.join(save_dir, 'test_metrics.csv')}")

if __name__ == "__main__":
    main()

