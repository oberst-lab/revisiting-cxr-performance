""" quantile_analysis_bow.py: Evaluates vision model performance across text-classifier quantile groups.
                             Uses pre-computed bag-of-words text-classifier quantiles to segment
                             and analyze vision model performance with bootstrap confidence intervals.

Usage:
python quantile_analysis_bow.py --label_index <index> --train_path <path> --val_path <path> --test_path <path> [--use_test_data]
"""
import torch
import numpy as np
import sys
import os
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, brier_score_loss
import torchmetrics
from tqdm import tqdm
import pickle
import argparse
from sklearn.utils import resample

def compute_weighted_ace(y_true, y_prob, weights=None, n_bins=10):
    """
    Computes Adaptive Calibration Error (ACE) with optional sample weights.
    """
    if weights is None:
        weights = np.ones_like(y_prob)

    idx = np.argsort(y_prob)
    y_prob = y_prob[idx]
    y_true = y_true[idx]
    weights = weights[idx]

    cum_weights = np.cumsum(weights)
    total_weight = cum_weights[-1]

    if total_weight <= 1e-8:
        return np.nan

    bin_edges = np.linspace(0, total_weight, n_bins + 1)

    ace = 0.0
    valid_bins = 0

    for i in range(n_bins):
        mask = (cum_weights > bin_edges[i]) & (cum_weights <= bin_edges[i+1])

        if not np.any(mask):
            continue

        w_bin = weights[mask]
        p_bin = y_prob[mask]
        y_bin = y_true[mask]

        bin_weight_sum = np.sum(w_bin)
        if bin_weight_sum <= 1e-12:
            continue

        avg_pred = np.average(p_bin, weights=w_bin)
        avg_true = np.average(y_bin, weights=w_bin)

        ace += np.abs(avg_true - avg_pred)
        valid_bins += 1

    return ace / valid_bins if valid_bins > 0 else 0.0

def compute_weighted_bss(y_true, y_prob, weights=None):
    """
    Computes Brier Score (BS) and Brier Skill Score (BSS) with optional sample weights.

    Returns:
        tuple: (bs, bss) - Brier Score and Brier Skill Score
    """
    if weights is None:
        weights = np.ones_like(y_prob)

    try:
        bs = brier_score_loss(y_true, y_prob, sample_weight=weights)
        prev = np.average(y_true, weights=weights)
        bs_ref = np.average((y_true - prev) ** 2, weights=weights)

        if bs_ref < 1e-8:
            return bs, 0.0

        bss = 1 - (bs / bs_ref)
        return bs, bss
    except Exception:
        return np.nan, np.nan

def compute_ci_percentiles(alpha=0.05, n_tests=13, bonferroni=True):
    """
    Compute confidence interval percentiles with optional Bonferroni correction.

    Args:
        alpha (float): Significance level (default: 0.05 for 95% CI)
        n_tests (int): Number of simultaneous tests for Bonferroni correction
        bonferroni (bool): Whether to apply Bonferroni correction

    Returns:
        tuple: (lower_percentile, upper_percentile)
    """
    if bonferroni:
        corrected_alpha = alpha / n_tests
    else:
        corrected_alpha = alpha

    lower_pct = (corrected_alpha / 2) * 100
    upper_pct = (1 - corrected_alpha / 2) * 100

    return lower_pct, upper_pct

def bootstrap_auroc_for_groups(label_index: int,
                               groups: dict,
                               num_bootstraps: int,
                               all_outputs: np.ndarray,
                               all_labels: np.ndarray,
                               bonferroni: bool = True,
                               n_tests: int = 13):
    """
    Perform bootstrapping to estimate AUROC, AUPRC, and Sensitivity@95%Specificity metrics for distinct groups.

    Args:
        label_index (int): The index of the label to evaluate.
        groups (dict): A dictionary where keys are group names (e.g., 'top_25')
                       and values are the corresponding sample indices.
        num_bootstraps (int): The number of bootstrap samples to generate.
        all_outputs (np.ndarray): The model predictions for the entire dataset.
        all_labels (np.ndarray): The true labels for the entire dataset.
        bonferroni (bool): Whether to apply Bonferroni correction for multiple testing (default: True)
        n_tests (int): Number of simultaneous tests for Bonferroni correction (default: 13)

    Returns:
        dict: A dictionary containing the mean and confidence intervals for
              AUROC, AUPRC, and Sensitivity@95%Specificity for each group.
    """
    bootstrapped_results = {group_name: {
        'auroc': [],
        'auprc': [],
        'sens_at_spec95': [],
        'sens_at_global_spec95': [],
        'spec_at_global_spec95': [],
        'ace': [],
        'bs': [],
        'bss': []
    } for group_name in groups.keys()}

    for _ in tqdm(range(num_bootstraps), desc='Bootstrapping Groups', unit='iter'):
        auroc_metric = torchmetrics.classification.MultilabelAUROC(num_labels=all_labels.shape[1], average='none')

        # First, collect all resampled data across groups to compute global threshold
        all_boot_outputs = []
        all_boot_labels = []
        group_boot_data = {}  # Store resampled data for each group

        for group_name, indices in groups.items():
            # Skip group if it has no samples
            if len(indices) == 0:
                group_boot_data[group_name] = None
                continue

            # Ensure there are at least two classes in the labels for this group
            group_labels = all_labels[indices]
            if len(np.unique(group_labels[:, label_index])) < 2:
                group_boot_data[group_name] = None
                continue

            # Resample the group with stratification
            try:
                boot_outputs, boot_labels = resample(
                    all_outputs[indices],
                    group_labels,
                    stratify=group_labels[:, label_index]
                )
                # Store resampled data for this group
                group_boot_data[group_name] = {
                    'outputs': boot_outputs,
                    'labels': boot_labels
                }
                # Collect for global threshold calculation
                all_boot_outputs.append(boot_outputs)
                all_boot_labels.append(boot_labels)
            except ValueError:  # Happens if a class has too few members for stratification
                group_boot_data[group_name] = None
                continue

        # Compute global threshold based on all groups combined
        global_threshold = None
        if len(all_boot_outputs) > 0:
            combined_outputs = np.vstack(all_boot_outputs)
            combined_labels = np.vstack(all_boot_labels)
            y_true_global = combined_labels[:, label_index]
            y_pred_global = combined_outputs[:, label_index]

            # Compute global ROC curve to find threshold at 95% specificity
            if len(np.unique(y_true_global)) >= 2:
                fpr_global, tpr_global, thresholds_global = roc_curve(y_true_global, y_pred_global)
                target_fpr = 0.05  # 95% specificity = 5% FPR
                idx_global = np.argmin(np.abs(fpr_global - target_fpr))
                global_threshold = thresholds_global[idx_global]

        # Now compute metrics for each group
        for group_name, boot_data in group_boot_data.items():
            if boot_data is None:
                bootstrapped_results[group_name]['auroc'].append(float('nan'))
                bootstrapped_results[group_name]['auprc'].append(float('nan'))
                bootstrapped_results[group_name]['sens_at_spec95'].append(float('nan'))
                bootstrapped_results[group_name]['sens_at_global_spec95'].append(float('nan'))
                bootstrapped_results[group_name]['spec_at_global_spec95'].append(float('nan'))
                bootstrapped_results[group_name]['ace'].append(float('nan'))
                bootstrapped_results[group_name]['bs'].append(float('nan'))
                bootstrapped_results[group_name]['bss'].append(float('nan'))
                continue

            boot_outputs = boot_data['outputs']
            boot_labels = boot_data['labels']

            # Compute AUROC for the resampled group
            auroc_metric.update(boot_outputs, boot_labels)
            auroc_val = auroc_metric.compute()[label_index]
            auroc_metric.reset()

            # Compute AUPRC for the resampled group
            y_true = boot_labels[:, label_index].cpu().numpy()
            y_pred = boot_outputs[:, label_index]
            auprc_val = average_precision_score(y_true, y_pred)

            # Compute Sensitivity @ 95% Specificity (subgroup-specific threshold)
            fpr, tpr, thresholds = roc_curve(y_true, y_pred)
            target_fpr = 0.05  # 95% specificity = 5% FPR
            idx = np.argmin(np.abs(fpr - target_fpr))
            sens_at_spec95_val = tpr[idx]

            # Compute Sensitivity and Specificity at global threshold
            if global_threshold is not None:
                # Ensure y_pred is numpy array
                y_pred_np = y_pred.cpu().numpy() if hasattr(y_pred, 'cpu') else np.array(y_pred)
                y_pred_binary = (y_pred_np >= global_threshold).astype(int)
                # Sensitivity = TP / (TP + FN)
                tp = np.sum((y_pred_binary == 1) & (y_true == 1))
                fn = np.sum((y_pred_binary == 0) & (y_true == 1))
                tn = np.sum((y_pred_binary == 0) & (y_true == 0))
                fp = np.sum((y_pred_binary == 1) & (y_true == 0))

                sens_global = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
                spec_global = tn / (tn + fp) if (tn + fp) > 0 else float('nan')
            else:
                sens_global = float('nan')
                spec_global = float('nan')

            # Compute calibration metrics
            # Convert to numpy if needed
            y_pred_np = y_pred.cpu().numpy() if hasattr(y_pred, 'cpu') else np.array(y_pred)
            ace_val = compute_weighted_ace(y_true, y_pred_np)
            bs_val, bss_val = compute_weighted_bss(y_true, y_pred_np)

            bootstrapped_results[group_name]['auroc'].append(auroc_val)
            bootstrapped_results[group_name]['auprc'].append(auprc_val)
            bootstrapped_results[group_name]['sens_at_spec95'].append(sens_at_spec95_val)
            bootstrapped_results[group_name]['sens_at_global_spec95'].append(sens_global)
            bootstrapped_results[group_name]['spec_at_global_spec95'].append(spec_global)
            bootstrapped_results[group_name]['ace'].append(ace_val)
            bootstrapped_results[group_name]['bs'].append(bs_val)
            bootstrapped_results[group_name]['bss'].append(bss_val)

    # Calculate final statistics for each group
    # Compute percentiles for confidence intervals (with optional Bonferroni correction)
    lower_pct, upper_pct = compute_ci_percentiles(alpha=0.05, n_tests=n_tests, bonferroni=bonferroni)

    final_results = {}
    for group_name, metrics_dict in bootstrapped_results.items():
        # Process AUROC
        valid_auroc = np.array([v for v in metrics_dict['auroc'] if not np.isnan(v)])
        if len(valid_auroc) > 0:
            mean_auroc = np.mean(valid_auroc)
            ci_auroc = np.percentile(valid_auroc, [lower_pct, upper_pct])
        else:
            mean_auroc = float('nan')
            ci_auroc = (float('nan'), float('nan'))

        # Process AUPRC
        valid_auprc = np.array([v for v in metrics_dict['auprc'] if not np.isnan(v)])
        if len(valid_auprc) > 0:
            mean_auprc = np.mean(valid_auprc)
            ci_auprc = np.percentile(valid_auprc, [lower_pct, upper_pct])
        else:
            mean_auprc = float('nan')
            ci_auprc = (float('nan'), float('nan'))

        # Process Sensitivity @ 95% Specificity (subgroup-specific threshold)
        valid_sens = np.array([v for v in metrics_dict['sens_at_spec95'] if not np.isnan(v)])
        if len(valid_sens) > 0:
            mean_sens = np.mean(valid_sens)
            ci_sens = np.percentile(valid_sens, [lower_pct, upper_pct])
        else:
            mean_sens = float('nan')
            ci_sens = (float('nan'), float('nan'))

        # Process Sensitivity @ 95% Specificity (global threshold)
        valid_sens_global = np.array([v for v in metrics_dict['sens_at_global_spec95'] if not np.isnan(v)])
        if len(valid_sens_global) > 0:
            mean_sens_global = np.mean(valid_sens_global)
            ci_sens_global = np.percentile(valid_sens_global, [lower_pct, upper_pct])
        else:
            mean_sens_global = float('nan')
            ci_sens_global = (float('nan'), float('nan'))

        # Process Specificity at global threshold
        valid_spec_global = np.array([v for v in metrics_dict['spec_at_global_spec95'] if not np.isnan(v)])
        if len(valid_spec_global) > 0:
            mean_spec_global = np.mean(valid_spec_global)
            ci_spec_global = np.percentile(valid_spec_global, [lower_pct, upper_pct])
        else:
            mean_spec_global = float('nan')
            ci_spec_global = (float('nan'), float('nan'))

        # Process ACE
        valid_ace = np.array([v for v in metrics_dict['ace'] if not np.isnan(v)])
        if len(valid_ace) > 0:
            mean_ace = np.mean(valid_ace)
            ci_ace = np.percentile(valid_ace, [lower_pct, upper_pct])
        else:
            mean_ace = float('nan')
            ci_ace = (float('nan'), float('nan'))

        # Process BS (Brier Score)
        valid_bs = np.array([v for v in metrics_dict['bs'] if not np.isnan(v)])
        if len(valid_bs) > 0:
            mean_bs = np.mean(valid_bs)
            ci_bs = np.percentile(valid_bs, [lower_pct, upper_pct])
        else:
            mean_bs = float('nan')
            ci_bs = (float('nan'), float('nan'))

        # Process BSS (Brier Skill Score)
        valid_bss = np.array([v for v in metrics_dict['bss'] if not np.isnan(v)])
        if len(valid_bss) > 0:
            mean_bss = np.mean(valid_bss)
            ci_bss = np.percentile(valid_bss, [lower_pct, upper_pct])
        else:
            mean_bss = float('nan')
            ci_bss = (float('nan'), float('nan'))

        final_results[group_name] = {
            'mean_auroc': mean_auroc,
            'auroc_ci': ci_auroc,
            'mean_auprc': mean_auprc,
            'auprc_ci': ci_auprc,
            'mean_sens_at_spec95': mean_sens,
            'sens_at_spec95_ci': ci_sens,
            'mean_sens_at_global_spec95': mean_sens_global,
            'sens_at_global_spec95_ci': ci_sens_global,
            'mean_spec_at_global_spec95': mean_spec_global,
            'spec_at_global_spec95_ci': ci_spec_global,
            'mean_ace': mean_ace,
            'ace_ci': ci_ace,
            'mean_bs': mean_bs,
            'bs_ci': ci_bs,
            'mean_bss': mean_bss,
            'bss_ci': ci_bss,
            'n_samples': len(groups[group_name])
        }

    return final_results


def main(args):
    # Load datasets
    print(f"[INFO] Loading datasets...")
    print(f"  - Train: {args.train_path}")
    print(f"  - Val: {args.val_path}")
    print(f"  - Test: {args.test_path}")
    
    train_df = pd.read_csv(args.train_path, compression='gzip')
    val_df = pd.read_csv(args.val_path, compression='gzip')
    test_df = pd.read_csv(args.test_path, compression='gzip')

    label_columns = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
                     'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 
                     'Lung Opacity', 'No Finding', 'Pleural Effusion', 
                     'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']
    
    if args.label_index < 0 or args.label_index >= len(label_columns):
        print(f"[ERROR] Label index {args.label_index} is out of range. Must be 0-{len(label_columns)-1}")
        sys.exit(1)
        
    label = label_columns[args.label_index]
    print(f"[INFO] Evaluating label: {label} (index {args.label_index})")

    # Determine output directory suffix based on Bonferroni correction usage
    bonferroni_suffix = "_bonferroni" if args.bonferroni else ""

    # Handle data selection
    if args.use_test_data:
        dataset = test_df
        predictions_dir = args.vision_predictions_test_dir
        quantile_dir = args.quantile_predictions_test_dir
        output_dir = args.output_test_dir or f"bow_quantile_auroc_evaluation_test_rbtl{bonferroni_suffix}"
        print(f"[INFO] Using test data")
    else:
        dataset = pd.concat([train_df, val_df])
        predictions_dir = args.vision_predictions_dir
        quantile_dir = args.quantile_predictions_dir
        output_dir = args.output_dir or f"bow_quantile_auroc_evaluation_rbtl{bonferroni_suffix}"
        print(f"[INFO] Using train + validation data")

    print(f"[INFO] Dataset size: {len(dataset)} samples")

    # Fill NAs and prepare labels
    dataset[label_columns] = dataset[label_columns].fillna(0)
    all_labels = torch.Tensor(dataset[label_columns].values).int()

    # Load vision model outputs
    predictions_path = f'{predictions_dir}/full_predictions.pkl'
    print(f"[INFO] Loading vision predictions from: {predictions_path}")
    if not os.path.exists(predictions_path):
        print(f"[ERROR] Vision predictions not found at: {predictions_path}")
        sys.exit(1)
    with open(predictions_path, 'rb') as f:
        all_outputs = pickle.load(f)

    # Load text-based quantile groups
    quantile_predictions_path = os.path.join(quantile_dir, label, f'{label}_quantile_predictions.pkl')
    print(f"[INFO] Loading quantile groups from: {quantile_predictions_path}")
    if not os.path.exists(quantile_predictions_path):
        print(f"[ERROR] Quantile predictions not found for label '{label}' at: {quantile_predictions_path}")
        print("[INFO] Please run the quantile generation script first.")
        sys.exit(1)
    with open(quantile_predictions_path, 'rb') as f:
        quantile_groups = pickle.load(f)

    print(f"[INFO] Loaded quantile groups for '{label}':")
    for group, indices in quantile_groups.items():
        print(f"  - {group}: {len(indices)} samples")

    # Prepare and run analysis
    num_bootstraps = args.num_bootstraps
    correction_msg = f"with Bonferroni correction (n_tests={args.n_tests})" if args.bonferroni else "without Bonferroni correction"
    print(f"[INFO] Starting bootstrap analysis with {num_bootstraps} iterations {correction_msg}...")

    results = bootstrap_auroc_for_groups(
        label_index=args.label_index,
        groups=quantile_groups,
        num_bootstraps=num_bootstraps,
        all_outputs=all_outputs,
        all_labels=all_labels,
        bonferroni=args.bonferroni,
        n_tests=args.n_tests
    )

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_path = f'{output_dir}/{label}_quantile_evaluations.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n[SUCCESS] Analysis complete for label '{label}'.")
    print("Results:")
    for group_name, data in results.items():
        print(f"  - {group_name} ({data['n_samples']} samples):")

        # AUROC
        if not np.isnan(data['mean_auroc']):
            print(f"    - AUROC: {data['mean_auroc']:.4f} (95% CI: {data['auroc_ci'][0]:.4f}, {data['auroc_ci'][1]:.4f})")
        else:
            print("    - AUROC: Could not compute")

        # AUPRC
        if not np.isnan(data['mean_auprc']):
            print(f"    - AUPRC: {data['mean_auprc']:.4f} (95% CI: {data['auprc_ci'][0]:.4f}, {data['auprc_ci'][1]:.4f})")
        else:
            print("    - AUPRC: Could not compute")

        # Sensitivity @ 95% Specificity (subgroup-specific threshold)
        if not np.isnan(data['mean_sens_at_spec95']):
            print(f"    - Sensitivity @ 95% Spec (subgroup threshold): {data['mean_sens_at_spec95']:.4f} (95% CI: {data['sens_at_spec95_ci'][0]:.4f}, {data['sens_at_spec95_ci'][1]:.4f})")
        else:
            print("    - Sensitivity @ 95% Spec (subgroup threshold): Could not compute")

        # Sensitivity @ 95% Specificity (global threshold)
        if not np.isnan(data['mean_sens_at_global_spec95']):
            print(f"    - Sensitivity @ 95% Spec (global threshold): {data['mean_sens_at_global_spec95']:.4f} (95% CI: {data['sens_at_global_spec95_ci'][0]:.4f}, {data['sens_at_global_spec95_ci'][1]:.4f})")
        else:
            print("    - Sensitivity @ 95% Spec (global threshold): Could not compute")

        # Specificity at global threshold
        if not np.isnan(data['mean_spec_at_global_spec95']):
            print(f"    - Specificity at global threshold: {data['mean_spec_at_global_spec95']:.4f} (95% CI: {data['spec_at_global_spec95_ci'][0]:.4f}, {data['spec_at_global_spec95_ci'][1]:.4f})")
        else:
            print("    - Specificity at global threshold: Could not compute")

        # ACE (Adaptive Calibration Error)
        if not np.isnan(data['mean_ace']):
            print(f"    - ACE (Adaptive Calibration Error): {data['mean_ace']:.4f} (95% CI: {data['ace_ci'][0]:.4f}, {data['ace_ci'][1]:.4f})")
        else:
            print("    - ACE: Could not compute")

        # BS (Brier Score)
        if not np.isnan(data['mean_bs']):
            print(f"    - BS (Brier Score): {data['mean_bs']:.4f} (95% CI: {data['bs_ci'][0]:.4f}, {data['bs_ci'][1]:.4f})")
        else:
            print("    - BS: Could not compute")

        # BSS (Brier Skill Score)
        if not np.isnan(data['mean_bss']):
            print(f"    - BSS (Brier Skill Score): {data['mean_bss']:.4f} (95% CI: {data['bss_ci'][0]:.4f}, {data['bss_ci'][1]:.4f})")
        else:
            print("    - BSS: Could not compute")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate vision model AUROC across text-based quantile groups.")
    
    # Required arguments
    parser.add_argument('--label_index', type=int, required=True, 
                       help='Index of the label to evaluate (0-13)')
    parser.add_argument('--train_path', type=str, required=True,
                       help='Path to training data CSV file')
    parser.add_argument('--val_path', type=str, required=True,
                       help='Path to validation data CSV file')
    parser.add_argument('--test_path', type=str, required=True,
                       help='Path to test data CSV file')
    
    # Directory arguments
    parser.add_argument('--vision_predictions_dir', type=str, 
                       default='vision_predictions',
                       help='Directory containing vision model predictions for train+val')
    parser.add_argument('--vision_predictions_test_dir', type=str,
                       default='vision_predictions_test', 
                       help='Directory containing vision model predictions for test')
    parser.add_argument('--quantile_predictions_dir', type=str,
                       default='bow_quantile_predictions',
                       help='Directory containing quantile predictions for train+val')
    parser.add_argument('--quantile_predictions_test_dir', type=str,
                       default='bow_quantile_predictions_test',
                       help='Directory containing quantile predictions for test')
    parser.add_argument('--output_dir', type=str,
                       help='Output directory for results (auto-determined if not specified)')
    parser.add_argument('--output_test_dir', type=str,
                       help='Output directory for test results (auto-determined if not specified)')
    
    # Optional arguments
    parser.add_argument('--use_test_data', action='store_true',
                       help='Use test data (and test predictions) instead of train+val')
    parser.add_argument('--num_bootstraps', type=int, default=10000,
                       help='Number of bootstrap iterations (default: 10000)')
    parser.add_argument('--no-bonferroni', dest='bonferroni',
                       action='store_false', default=True,
                       help='Disable Bonferroni correction (default: enabled)')
    parser.add_argument('--n_tests', type=int, default=13,
                       help='Number of simultaneous tests for Bonferroni correction (default: 13)')

    args = parser.parse_args()
    
    main(args)