import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
import pandas as pd
import pickle
import os
from tqdm import tqdm
import argparse
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.utils import resample
from sklearn.calibration import CalibratedClassifierCV

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

def nearest_neighbor_matching(data_df, label_column, text_probs):
    """Perform 1:1 matching between positive and negative cases based on text probabilities"""
    pos_indices = data_df[data_df[label_column] == 1].index.values
    neg_indices = data_df[data_df[label_column] == 0].index.values
    if len(pos_indices) == 0 or len(neg_indices) == 0:
        return {"positive": [], "negative": []}
    pos_probs = text_probs[pos_indices].reshape(-1, 1)
    neg_probs = text_probs[neg_indices].reshape(-1, 1)
    cost_matrix = np.abs(pos_probs - neg_probs.T)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    if len(pos_indices) > len(neg_indices):
        row_ind = row_ind[:len(neg_indices)]
        col_ind = col_ind[:len(neg_indices)]
    return {
        "positive": pos_indices[row_ind],
        "negative": neg_indices[col_ind]
    }

def stratified_bootstrap_resample(labels, random_state=None):
    """Perform stratified bootstrap resampling while maintaining class balance"""
    unique_classes, counts = np.unique(labels, return_counts=True)
    sample_indices = []
    
    for class_label, count in zip(unique_classes, counts):
        class_indices = np.where(labels == class_label)[0]
        bootstrap_indices = resample(class_indices,
                                   replace=True,
                                   n_samples=count,
                                   random_state=random_state)
        sample_indices.extend(bootstrap_indices)
    
    return np.array(sample_indices)

def bootstrap_auroc_comparison(
    label_index,
    all_outputs,
    all_labels,
    text_probs,
    data_df,
    label_columns,
    num_bootstraps=100000,
    bonferroni=True,
    n_tests=13
):
    """Bootstrap vision and matched neighbor metrics (AUROC, AUPRC, Sensitivity@95%Spec) and their differences"""
    vision_metrics = {
        'auroc': [],
        'auprc': [],
        'sens_at_spec95': [],
        'sens_at_global_spec95': [],
        'spec_at_global_spec95': []
    }
    matched_metrics = {
        'auroc': [],
        'auprc': [],
        'sens_at_spec95': [],
        'sens_at_global_spec95': [],
        'spec_at_global_spec95': []
    }
    diffs = {
        'auroc': [],
        'auprc': [],
        'sens_at_spec95': [],
        'sens_at_global_spec95': [],
        'spec_at_global_spec95': []
    }
    n_matches = []

    current_labels = all_labels[:, label_index]
    
    for i in tqdm(range(num_bootstraps), desc="Bootstrapping"):
        # Stratified resampling
        boot_indices = stratified_bootstrap_resample(current_labels, random_state=i)
        boot_df = data_df.iloc[boot_indices].reset_index(drop=True)
        boot_text_probs = text_probs[boot_indices]
        boot_vision_probs = all_outputs[boot_indices, label_index]
        boot_labels = current_labels[boot_indices]
        
        # Skip if no class diversity
        if len(np.unique(boot_labels)) < 2:
            continue
            
        # Calculate vision model metrics
        try:
            vision_auroc = roc_auc_score(boot_labels, boot_vision_probs)
            vision_auprc = average_precision_score(boot_labels, boot_vision_probs)

            # Compute Sensitivity @ 95% Specificity for vision (subgroup-specific)
            fpr_v, tpr_v, thresholds_v = roc_curve(boot_labels, boot_vision_probs)
            target_fpr = 0.05
            idx_v = np.argmin(np.abs(fpr_v - target_fpr))
            vision_sens = tpr_v[idx_v]

            # Compute global threshold based on entire bootstrap sample
            global_threshold = thresholds_v[idx_v]

            # Compute Sensitivity and Specificity at global threshold for vision
            vision_probs_np = boot_vision_probs if isinstance(boot_vision_probs, np.ndarray) else boot_vision_probs.cpu().numpy()
            vision_labels_np = boot_labels if isinstance(boot_labels, np.ndarray) else boot_labels.cpu().numpy()
            y_pred_binary_v = (vision_probs_np >= global_threshold).astype(int)

            tp_v = np.sum((y_pred_binary_v == 1) & (vision_labels_np == 1))
            fn_v = np.sum((y_pred_binary_v == 0) & (vision_labels_np == 1))
            tn_v = np.sum((y_pred_binary_v == 0) & (vision_labels_np == 0))
            fp_v = np.sum((y_pred_binary_v == 1) & (vision_labels_np == 0))

            vision_sens_global = tp_v / (tp_v + fn_v) if (tp_v + fn_v) > 0 else np.nan
            vision_spec_global = tn_v / (tn_v + fp_v) if (tn_v + fp_v) > 0 else np.nan
        except ValueError:
            continue

        # Get matched pairs
        matched_pairs = nearest_neighbor_matching(
            boot_df,
            label_columns[label_index],
            boot_text_probs
        )

        if len(matched_pairs["positive"]) == 0:
            continue

        # Calculate matched metrics
        try:
            matched_labels = np.concatenate([
                boot_labels[matched_pairs["positive"]],
                boot_labels[matched_pairs["negative"]]
            ])
            matched_probs = np.concatenate([
                boot_vision_probs[matched_pairs["positive"]],
                boot_vision_probs[matched_pairs["negative"]]
            ])
            matched_auroc = roc_auc_score(matched_labels, matched_probs)
            matched_auprc = average_precision_score(matched_labels, matched_probs)

            # Compute Sensitivity @ 95% Specificity for matched (subgroup-specific)
            fpr_m, tpr_m, _ = roc_curve(matched_labels, matched_probs)
            idx_m = np.argmin(np.abs(fpr_m - target_fpr))
            matched_sens = tpr_m[idx_m]

            # Compute Sensitivity and Specificity at global threshold for matched
            matched_probs_np = matched_probs if isinstance(matched_probs, np.ndarray) else matched_probs.cpu().numpy()
            matched_labels_np = matched_labels if isinstance(matched_labels, np.ndarray) else matched_labels.cpu().numpy()
            y_pred_binary_m = (matched_probs_np >= global_threshold).astype(int)

            tp_m = np.sum((y_pred_binary_m == 1) & (matched_labels_np == 1))
            fn_m = np.sum((y_pred_binary_m == 0) & (matched_labels_np == 1))
            tn_m = np.sum((y_pred_binary_m == 0) & (matched_labels_np == 0))
            fp_m = np.sum((y_pred_binary_m == 1) & (matched_labels_np == 0))

            matched_sens_global = tp_m / (tp_m + fn_m) if (tp_m + fn_m) > 0 else np.nan
            matched_spec_global = tn_m / (tn_m + fp_m) if (tn_m + fp_m) > 0 else np.nan

            vision_metrics['auroc'].append(vision_auroc)
            vision_metrics['auprc'].append(vision_auprc)
            vision_metrics['sens_at_spec95'].append(vision_sens)
            vision_metrics['sens_at_global_spec95'].append(vision_sens_global)
            vision_metrics['spec_at_global_spec95'].append(vision_spec_global)

            matched_metrics['auroc'].append(matched_auroc)
            matched_metrics['auprc'].append(matched_auprc)
            matched_metrics['sens_at_spec95'].append(matched_sens)
            matched_metrics['sens_at_global_spec95'].append(matched_sens_global)
            matched_metrics['spec_at_global_spec95'].append(matched_spec_global)

            diffs['auroc'].append(vision_auroc - matched_auroc)
            diffs['auprc'].append(vision_auprc - matched_auprc)
            diffs['sens_at_spec95'].append(vision_sens - matched_sens)
            diffs['sens_at_global_spec95'].append(vision_sens_global - matched_sens_global)
            diffs['spec_at_global_spec95'].append(vision_spec_global - matched_spec_global)

            n_matches.append(len(matched_pairs["positive"]))
        except ValueError:
            continue
    
    # Handle empty results
    if not vision_metrics['auroc']:
        print(f"Warning: No valid bootstrap samples for {data_df.columns[label_index]}")
        return {
            'vision': {
                'auroc': {'mean': np.nan, 'ci': [np.nan, np.nan]},
                'auprc': {'mean': np.nan, 'ci': [np.nan, np.nan]},
                'sens_at_spec95': {'mean': np.nan, 'ci': [np.nan, np.nan]},
                'sens_at_global_spec95': {'mean': np.nan, 'ci': [np.nan, np.nan]},
                'spec_at_global_spec95': {'mean': np.nan, 'ci': [np.nan, np.nan]}
            },
            'matched': {
                'auroc': {'mean': np.nan, 'ci': [np.nan, np.nan]},
                'auprc': {'mean': np.nan, 'ci': [np.nan, np.nan]},
                'sens_at_spec95': {'mean': np.nan, 'ci': [np.nan, np.nan]},
                'sens_at_global_spec95': {'mean': np.nan, 'ci': [np.nan, np.nan]},
                'spec_at_global_spec95': {'mean': np.nan, 'ci': [np.nan, np.nan]}
            },
            'difference': {
                'auroc': {'mean': np.nan, 'ci': [np.nan, np.nan]},
                'auprc': {'mean': np.nan, 'ci': [np.nan, np.nan]},
                'sens_at_spec95': {'mean': np.nan, 'ci': [np.nan, np.nan]},
                'sens_at_global_spec95': {'mean': np.nan, 'ci': [np.nan, np.nan]},
                'spec_at_global_spec95': {'mean': np.nan, 'ci': [np.nan, np.nan]}
            },
            'avg_matches': np.nan,
            'n_bootstraps': 0
        }

    # Calculate summary statistics
    # Compute percentiles for confidence intervals (with optional Bonferroni correction)
    lower_pct, upper_pct = compute_ci_percentiles(alpha=0.05, n_tests=n_tests, bonferroni=bonferroni)

    results = {
        'vision': {
            'auroc': {
                'mean': np.mean(vision_metrics['auroc']),
                'ci': np.percentile(vision_metrics['auroc'], [lower_pct, upper_pct])
            },
            'auprc': {
                'mean': np.mean(vision_metrics['auprc']),
                'ci': np.percentile(vision_metrics['auprc'], [lower_pct, upper_pct])
            },
            'sens_at_spec95': {
                'mean': np.mean(vision_metrics['sens_at_spec95']),
                'ci': np.percentile(vision_metrics['sens_at_spec95'], [lower_pct, upper_pct])
            },
            'sens_at_global_spec95': {
                'mean': np.mean(vision_metrics['sens_at_global_spec95']),
                'ci': np.percentile(vision_metrics['sens_at_global_spec95'], [lower_pct, upper_pct])
            },
            'spec_at_global_spec95': {
                'mean': np.mean(vision_metrics['spec_at_global_spec95']),
                'ci': np.percentile(vision_metrics['spec_at_global_spec95'], [lower_pct, upper_pct])
            }
        },
        'matched': {
            'auroc': {
                'mean': np.mean(matched_metrics['auroc']),
                'ci': np.percentile(matched_metrics['auroc'], [lower_pct, upper_pct])
            },
            'auprc': {
                'mean': np.mean(matched_metrics['auprc']),
                'ci': np.percentile(matched_metrics['auprc'], [lower_pct, upper_pct])
            },
            'sens_at_spec95': {
                'mean': np.mean(matched_metrics['sens_at_spec95']),
                'ci': np.percentile(matched_metrics['sens_at_spec95'], [lower_pct, upper_pct])
            },
            'sens_at_global_spec95': {
                'mean': np.mean(matched_metrics['sens_at_global_spec95']),
                'ci': np.percentile(matched_metrics['sens_at_global_spec95'], [lower_pct, upper_pct])
            },
            'spec_at_global_spec95': {
                'mean': np.mean(matched_metrics['spec_at_global_spec95']),
                'ci': np.percentile(matched_metrics['spec_at_global_spec95'], [lower_pct, upper_pct])
            }
        },
        'difference': {
            'auroc': {
                'mean': np.mean(diffs['auroc']),
                'ci': np.percentile(diffs['auroc'], [lower_pct, upper_pct])
            },
            'auprc': {
                'mean': np.mean(diffs['auprc']),
                'ci': np.percentile(diffs['auprc'], [lower_pct, upper_pct])
            },
            'sens_at_spec95': {
                'mean': np.mean(diffs['sens_at_spec95']),
                'ci': np.percentile(diffs['sens_at_spec95'], [lower_pct, upper_pct])
            },
            'sens_at_global_spec95': {
                'mean': np.mean(diffs['sens_at_global_spec95']),
                'ci': np.percentile(diffs['sens_at_global_spec95'], [lower_pct, upper_pct])
            },
            'spec_at_global_spec95': {
                'mean': np.mean(diffs['spec_at_global_spec95']),
                'ci': np.percentile(diffs['spec_at_global_spec95'], [lower_pct, upper_pct])
            }
        },
        'avg_matches': np.mean(n_matches),
        'n_bootstraps': len(vision_metrics['auroc'])
    }

    return results

def main(args):
    # Load data
    if args.use_test_data:
        dataset = pd.read_csv(args.test_csv, compression='gzip')
        embeddings_path = args.test_embeddings
        predictions_dir = args.vision_predictions_test
    else:
        train_df = pd.read_csv(args.train_csv, compression='gzip')
        val_df = pd.read_csv(args.val_csv, compression='gzip')
        dataset = pd.concat([train_df, val_df])
        embeddings_path = args.train_val_embeddings
        predictions_dir = args.vision_predictions_train_val
    
    label_columns = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
                    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 
                    'Lung Opacity', 'No Finding', 'Pleural Effusion', 
                    'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']
    
    dataset[label_columns] = dataset[label_columns].fillna(0)
    all_labels = torch.Tensor(dataset[label_columns].values).int()

    # Load vision model outputs
    with open(f'{predictions_dir}/full_predictions.pkl', 'rb') as f:
        all_outputs = pickle.load(f)

    # --- Get Text Model Probabilities (Embedding-based) ---
    # Load the consolidated metrics file
    if not os.path.exists(args.metrics_file):
        print(f"[ERROR] Metrics file not found at {args.metrics_file}. Exiting.")
        return
        
    metrics_df = pd.read_csv(args.metrics_file)
    best_df = metrics_df.sort_values(by='AUROC', ascending=False).drop_duplicates('label_name').copy()
    
    label_name = label_columns[args.label_index]
    clf_name = str(best_df[best_df["label_name"] == label_name]["classifier_name"].values[0])
    model_path = os.path.join(args.text_models_dir, f"{label_name}_{clf_name}_train_val_split.pkl")

    # Load text embeddings
    embeddings = torch.load(embeddings_path, weights_only=False)
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.numpy()

    # Load and apply text classifier
    with open(model_path, "rb") as f:
        clf = pickle.load(f)

    if not hasattr(clf, "predict_proba"):
        calibrated_clf = CalibratedClassifierCV(clf, method="sigmoid", cv="prefit")
        calibrated_clf.fit(embeddings, dataset[label_name].values)
        text_probs = calibrated_clf.predict_proba(embeddings)[:, 1]
    else:
        text_probs = clf.predict_proba(embeddings)[:, 1]

    # Run bootstrap analysis
    correction_msg = f"with Bonferroni correction (n_tests={args.n_tests})" if args.bonferroni else "without Bonferroni correction"
    print(f"[INFO] Starting bootstrap analysis with {args.num_bootstraps} iterations {correction_msg}...")

    results = bootstrap_auroc_comparison(
        label_index=args.label_index,
        all_outputs=all_outputs,
        all_labels=all_labels.numpy(),
        text_probs=text_probs,
        data_df=dataset,
        label_columns=label_columns,
        num_bootstraps=args.num_bootstraps,
        bonferroni=args.bonferroni,
        n_tests=args.n_tests
    )

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    with open(f'{args.output_dir}/{label_name}_comparison.pkl', 'wb') as f:
        pickle.dump(results, f)

    print(f"\nResults for {label_name}:")
    print("Vision Model:")
    print(f"  AUROC: {results['vision']['auroc']['mean']:.3f} ({results['vision']['auroc']['ci'][0]:.3f}-{results['vision']['auroc']['ci'][1]:.3f})")
    print(f"  AUPRC: {results['vision']['auprc']['mean']:.3f} ({results['vision']['auprc']['ci'][0]:.3f}-{results['vision']['auprc']['ci'][1]:.3f})")
    print(f"  Sensitivity @ 95% Spec (subgroup threshold): {results['vision']['sens_at_spec95']['mean']:.3f} ({results['vision']['sens_at_spec95']['ci'][0]:.3f}-{results['vision']['sens_at_spec95']['ci'][1]:.3f})")
    print(f"  Sensitivity @ 95% Spec (global threshold): {results['vision']['sens_at_global_spec95']['mean']:.3f} ({results['vision']['sens_at_global_spec95']['ci'][0]:.3f}-{results['vision']['sens_at_global_spec95']['ci'][1]:.3f})")
    print(f"  Specificity at global threshold: {results['vision']['spec_at_global_spec95']['mean']:.3f} ({results['vision']['spec_at_global_spec95']['ci'][0]:.3f}-{results['vision']['spec_at_global_spec95']['ci'][1]:.3f})")
    print("Matched Neighbor:")
    print(f"  AUROC: {results['matched']['auroc']['mean']:.3f} ({results['matched']['auroc']['ci'][0]:.3f}-{results['matched']['auroc']['ci'][1]:.3f})")
    print(f"  AUPRC: {results['matched']['auprc']['mean']:.3f} ({results['matched']['auprc']['ci'][0]:.3f}-{results['matched']['auprc']['ci'][1]:.3f})")
    print(f"  Sensitivity @ 95% Spec (subgroup threshold): {results['matched']['sens_at_spec95']['mean']:.3f} ({results['matched']['sens_at_spec95']['ci'][0]:.3f}-{results['matched']['sens_at_spec95']['ci'][1]:.3f})")
    print(f"  Sensitivity @ 95% Spec (global threshold): {results['matched']['sens_at_global_spec95']['mean']:.3f} ({results['matched']['sens_at_global_spec95']['ci'][0]:.3f}-{results['matched']['sens_at_global_spec95']['ci'][1]:.3f})")
    print(f"  Specificity at global threshold: {results['matched']['spec_at_global_spec95']['mean']:.3f} ({results['matched']['spec_at_global_spec95']['ci'][0]:.3f}-{results['matched']['spec_at_global_spec95']['ci'][1]:.3f})")
    print("Difference (Vision - Matched):")
    print(f"  AUROC Diff: {results['difference']['auroc']['mean']:.3f} ({results['difference']['auroc']['ci'][0]:.3f}-{results['difference']['auroc']['ci'][1]:.3f})")
    print(f"  AUPRC Diff: {results['difference']['auprc']['mean']:.3f} ({results['difference']['auprc']['ci'][0]:.3f}-{results['difference']['auprc']['ci'][1]:.3f})")
    print(f"  Sensitivity Diff (subgroup threshold): {results['difference']['sens_at_spec95']['mean']:.3f} ({results['difference']['sens_at_spec95']['ci'][0]:.3f}-{results['difference']['sens_at_spec95']['ci'][1]:.3f})")
    print(f"  Sensitivity Diff (global threshold): {results['difference']['sens_at_global_spec95']['mean']:.3f} ({results['difference']['sens_at_global_spec95']['ci'][0]:.3f}-{results['difference']['sens_at_global_spec95']['ci'][1]:.3f})")
    print(f"  Specificity Diff (global threshold): {results['difference']['spec_at_global_spec95']['mean']:.3f} ({results['difference']['spec_at_global_spec95']['ci'][0]:.3f}-{results['difference']['spec_at_global_spec95']['ci'][1]:.3f})")
    print(f"Average matches per bootstrap: {results['avg_matches']:.1f}")
    print(f"Successful bootstraps: {results['n_bootstraps']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare vision and text model performance using matched neighbors")
    
    # Required arguments
    parser.add_argument('--label_index', type=int, required=True,
                       help='Index of the label to analyze (0-13)')
    
    # Data file paths
    parser.add_argument('--test_csv', type=str, 
                       default='experiments/experiments.test.csv.gz',
                       help='Path to test CSV file')
    parser.add_argument('--train_csv', type=str,
                       default='experiments/train.csv.gz', 
                       help='Path to train CSV file')
    parser.add_argument('--val_csv', type=str,
                       default='experiments/val.csv.gz',
                       help='Path to validation CSV file')
    
    # Embedding paths
    parser.add_argument('--test_embeddings', type=str,
                       default='experiments/test.pt',
                       help='Path to test embeddings file')
    parser.add_argument('--train_val_embeddings', type=str,
                       default='experiments/train_val.pt',
                       help='Path to train+val embeddings file')
    
    # Model and prediction paths
    parser.add_argument('--vision_predictions_test', type=str,
                       default='vision_predictions_test',
                       help='Directory containing test vision predictions')
    parser.add_argument('--vision_predictions_train_val', type=str,
                       default='vision_predictions',
                       help='Directory containing train+val vision predictions')
    parser.add_argument('--metrics_file', type=str,
                       default='text_classifiers_metrics_train_val_split/all_results.csv',
                       help='Path to text classifier metrics file')
    parser.add_argument('--text_models_dir', type=str,
                       default='text_models',
                       help='Directory containing text model files')
    
    # Output settings
    parser.add_argument('--output_dir', type=str,
                       help='Output directory for results (auto-determined if not specified)')
    
    # Analysis parameters
    parser.add_argument('--use_test_data', action='store_true',
                       help='Use test data instead of train+val data')
    parser.add_argument('--num_bootstraps', type=int, default=10000,
                       help='Number of bootstrap iterations')
    parser.add_argument('--no-bonferroni', dest='bonferroni',
                       action='store_false', default=True,
                       help='Disable Bonferroni correction (default: enabled)')
    parser.add_argument('--n_tests', type=int, default=13,
                       help='Number of simultaneous tests for Bonferroni correction (default: 13)')

    args = parser.parse_args()

    # Auto-determine output directory if not specified
    if args.output_dir is None:
        bonferroni_suffix = "_bonferroni" if args.bonferroni else ""
        if args.use_test_data:
            args.output_dir = f"matched_neighbor_results_test_rbtl{bonferroni_suffix}"
        else:
            args.output_dir = f"matched_neighbor_results_rbtl{bonferroni_suffix}"

    main(args)