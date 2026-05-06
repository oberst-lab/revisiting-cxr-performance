import torch
import numpy as np
import sys
import os
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import torchmetrics
from tqdm import tqdm
import pickle
import argparse
from sklearn.utils import resample

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
        'spec_at_global_spec95': []
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

            bootstrapped_results[group_name]['auroc'].append(auroc_val)
            bootstrapped_results[group_name]['auprc'].append(auprc_val)
            bootstrapped_results[group_name]['sens_at_spec95'].append(sens_at_spec95_val)
            bootstrapped_results[group_name]['sens_at_global_spec95'].append(sens_global)
            bootstrapped_results[group_name]['spec_at_global_spec95'].append(spec_global)

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
            'n_samples': len(groups[group_name])
        }

    return final_results


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate vision model AUROC across text-based quantile groups.")
    parser.add_argument('--label_index', type=int, required=True, 
                       help='Index of the label to evaluate (0-13)')
    parser.add_argument('--train_path', type=str, required=True,
                       help='Path to training data CSV file')
    parser.add_argument('--val_path', type=str, required=True,
                       help='Path to validation data CSV file')
    parser.add_argument('--test_path', type=str, required=True,
                       help='Path to test data CSV file')
    parser.add_argument('--use_test_data', action='store_true', 
                       help='Use test data (and test predictions) instead of train+val')
    parser.add_argument('--predictions_dir', type=str, default=None,
                       help='Directory containing vision predictions (auto-determined if not specified)')
    parser.add_argument('--quantile_dir', type=str, default=None,
                       help='Directory containing quantile predictions (auto-determined if not specified)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results (auto-determined if not specified)')
    parser.add_argument('--num_bootstraps', type=int, default=10000,
                       help='Number of bootstrap samples (default: 10000)')
    parser.add_argument('--no-bonferroni', dest='bonferroni',
                       action='store_false', default=True,
                       help='Disable Bonferroni correction (default: enabled)')
    parser.add_argument('--n_tests', type=int, default=13,
                       help='Number of simultaneous tests for Bonferroni correction (default: 13)')

    args = parser.parse_args()
    
    # Validate file paths
    for path_name, path in [('train_path', args.train_path), 
                           ('val_path', args.val_path), 
                           ('test_path', args.test_path)]:
        if not os.path.exists(path):
            print(f"[ERROR] {path_name} does not exist: {path}")
            sys.exit(1)
    
    # Load datasets
    print(f"[INFO] Loading training data from: {args.train_path}")
    train_df = pd.read_csv(args.train_path, compression='gzip' if args.train_path.endswith('.gz') else None)
    
    print(f"[INFO] Loading validation data from: {args.val_path}")
    val_df = pd.read_csv(args.val_path, compression='gzip' if args.val_path.endswith('.gz') else None)
    
    print(f"[INFO] Loading test data from: {args.test_path}")
    test_df = pd.read_csv(args.test_path, compression='gzip' if args.test_path.endswith('.gz') else None)

    label_columns = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
                     'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 
                     'Lung Opacity', 'No Finding', 'Pleural Effusion', 
                     'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']
    
    # Validate label index
    if args.label_index < 0 or args.label_index >= len(label_columns):
        print(f"[ERROR] label_index must be between 0 and {len(label_columns)-1}")
        sys.exit(1)
    
    label = label_columns[args.label_index]
    print(f"[INFO] Evaluating label: {label} (index: {args.label_index})")

    # Handle data selection and directory paths
    # Determine output directory suffix based on Bonferroni correction usage
    bonferroni_suffix = "_bonferroni" if args.bonferroni else ""

    if args.use_test_data:
        dataset = test_df
        predictions_dir = args.predictions_dir or "vision_predictions_test"
        quantile_dir = args.quantile_dir or "quantile_predictions_test"
        output_dir = args.output_dir or f"quantile_auroc_evaluation_test_rbtl{bonferroni_suffix}"
        print(f"[INFO] Using test dataset ({len(dataset)} samples)")
    else:
        dataset = pd.concat([train_df, val_df])
        predictions_dir = args.predictions_dir or "vision_predictions"
        quantile_dir = args.quantile_dir or "quantile_predictions"
        output_dir = args.output_dir or f"quantile_auroc_evaluation_rbtl{bonferroni_suffix}"
        print(f"[INFO] Using combined train+val dataset ({len(dataset)} samples)")

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
    correction_msg = f"with Bonferroni correction (n_tests={args.n_tests})" if args.bonferroni else "without Bonferroni correction"
    print(f"[INFO] Starting bootstrap analysis with {args.num_bootstraps} iterations {correction_msg}...")

    results = bootstrap_auroc_for_groups(
        label_index=args.label_index,
        groups=quantile_groups,
        num_bootstraps=args.num_bootstraps,
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

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()