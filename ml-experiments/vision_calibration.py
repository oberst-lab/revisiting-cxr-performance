"""
vision_calibration_analysis.py: 
Computes calibration metrics (ACE, Brier Score) and generates calibration plots
for the calibrated Vision Model predictions on the Test set.
"""

import pandas as pd
import numpy as np
import os
import argparse
import sys
import pickle
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.calibration import calibration_curve

# --- CONFIGURATION ---
# Set plot style for academic papers
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 14})

def compute_ace(y_true, y_prob, n_bins=10):
    """
    Computes Adaptive Calibration Error (ACE) using quantile binning.
    """
    # 1. Sort by probability
    idx = np.argsort(y_prob)
    y_prob_sorted = y_prob[idx]
    y_true_sorted = y_true[idx]
    
    # 2. Split into N equal-frequency bins
    sub_probs = np.array_split(y_prob_sorted, n_bins)
    sub_labels = np.array_split(y_true_sorted, n_bins)
    
    ace = 0.0
    valid_bins = 0
    
    for chunk_probs, chunk_labels in zip(sub_probs, sub_labels):
        if len(chunk_labels) == 0: continue
        
        bin_conf = np.mean(chunk_probs)
        bin_acc = np.mean(chunk_labels)
        ace += np.abs(bin_acc - bin_conf)
        valid_bins += 1
        
    return ace / valid_bins if valid_bins > 0 else 0.0

def compute_brier_skill_score(y_true, y_prob):
    """
    Computes Brier Score and Brier Skill Score (BSS).
    """
    bs = metrics.brier_score_loss(y_true, y_prob)
    
    # Baseline: Predict the prevalence for everyone
    prevalence = np.mean(y_true)
    bs_baseline = np.mean((y_true - prevalence) ** 2)
    
    if bs_baseline < 1e-8:
        bss = 0.0
    else:
        bss = 1 - (bs / bs_baseline)
        
    return bs, bss

def plot_calibration(y_true, y_prob, label_name, model_name, ace, bss, output_path):
    """
    Generates and saves a calibration plot using Quantile strategy.
    """
    # Use quantile strategy for the plot points to match ACE logic
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='quantile')

    plt.figure(figsize=(8, 8))
    
    # 1. Perfectly Calibrated Line
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    
    # 2. Model Performance
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, color='darkorange', label=f'{model_name}')
    
    # 3. Density Histogram (at the bottom)
    plt.hist(y_prob, range=(0, 1), bins=50, density=True, color='orange', alpha=0.2, 
             histtype='stepfilled', label='Prediction Distribution')

    # Formatting
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives (Observed)')
    plt.title(f'Calibration: {label_name} (Vision Only)')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Add Text Box with Metrics
    textstr = '\n'.join((
        f'ACE: {ace:.4f}',
        f'Brier Skill: {bss:.4f}'
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    plt.text(0.95, 0.05, textstr, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='bottom', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def main(args):
    # Setup Directories
    output_dir = args.output_dir
    plots_dir = os.path.join(output_dir, 'vision_calibration_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    label_columns = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
                     'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 
                     'Lung Opacity', 'Pleural Effusion', 
                     'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']

    # 1. Load Test Labels (Ground Truth)
    print("[INFO] Loading Test Labels CSV...")
    try:
        df_test = pd.read_csv(args.test_csv, compression='gzip' if args.test_csv.endswith('.gz') else None)
        df_test[label_columns] = df_test[label_columns].fillna(0)
    except Exception as e:
        print(f"[ERROR] Failed to load Test CSV: {e}")
        sys.exit(1)

    # 2. Load Vision Predictions
    # This expects the .pkl file output from your vision_evaluation.py script
    # It contains a Tensor of shape (N_samples, 14)
    print(f"[INFO] Loading Vision Predictions from {args.predictions_pkl}...")
    try:
        with open(args.predictions_pkl, "rb") as f:
            # Depending on how it was saved, it might be a Tensor or a dict
            # Your script saves: all_outputs (Tensor)
            # Or if you used the merging script: {'outputs': ..., 'labels': ...}
            data = pickle.load(f)
            
            if isinstance(data, dict):
                # If loaded from merged dict
                predictions = data['outputs']
                # Verify labels align if possible (optional)
            elif torch.is_tensor(data):
                predictions = data
            elif isinstance(data, list): # rare case
                predictions = torch.cat(data)
            else:
                print(f"[ERROR] Unknown data format in pickle: {type(data)}")
                sys.exit(1)
                
            # Convert to numpy
            if torch.is_tensor(predictions):
                predictions = predictions.numpy()
                
    except Exception as e:
        print(f"[ERROR] Failed to load predictions pickle: {e}")
        sys.exit(1)

    # Sanity Check
    if predictions.shape[0] != len(df_test):
        print(f"[ERROR] Mismatch! Test set has {len(df_test)} rows, but predictions have {predictions.shape[0]} rows.")
        sys.exit(1)

    metrics_results = []
    
    print("\n[INFO] Computing Calibration Metrics per Label...")
    
    for i, label in enumerate(label_columns):
        print(f"   -> Processing {label}...")
        
        y_true = df_test[label].values
        y_prob = predictions[:, i] # Column i corresponds to this label
        
        # A. Calculate Metrics
        ace = compute_ace(y_true, y_prob, n_bins=10)
        bs, bss = compute_brier_skill_score(y_true, y_prob)
        auroc = metrics.roc_auc_score(y_true, y_prob)
        
        # B. Generate Plot
        plot_filename = f"calibration_vision_{label.replace(' ', '_')}.png"
        plot_path = os.path.join(plots_dir, plot_filename)
        
        plot_calibration(y_true, y_prob, label, "DenseNet121 (Vision)", ace, bss, plot_path)
        
        metrics_results.append({
            'label_name': label,
            'model_type': 'Vision_Only',
            'ACE': ace,
            'Brier_Score': bs,
            'Brier_Skill_Score': bss,
            'AUROC_Test': auroc
        })

    # 3. Save Summary CSV
    if metrics_results:
        results_df = pd.DataFrame(metrics_results)
        csv_path = os.path.join(output_dir, 'vision_calibration_metrics.csv')
        results_df.to_csv(csv_path, index=False)
        
        print("\n" + "="*60)
        print("VISION MODEL CALIBRATION SUMMARY")
        print("="*60)
        print(results_df[['label_name', 'ACE', 'Brier_Skill_Score', 'AUROC_Test']].round(4).to_string(index=False))
        print(f"\nSaved plots to: {plots_dir}")
        print(f"Saved metrics to: {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_csv', type=str, required=True, 
                        help='Path to test.csv.gz (Ground Truth)')
    parser.add_argument('--predictions_pkl', type=str, required=True, 
                        help='Path to the .pkl file containing calibrated vision predictions')
    parser.add_argument('--output_dir', type=str, default='vision_analysis_results', 
                        help='Directory to save plots and csvs')
    args = parser.parse_args()
    main(args)