import json
import re
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

PATHOLOGY_KEYWORDS = {
    0: {  # Atelectasis
        'keywords': ['atelectasis'],
        'label_name': 'Atelectasis'
    },
    1: {  # Cardiomegaly  
        'keywords': ['cardiomegaly'],
        'label_name': 'Cardiomegaly'
    },
    2: {  # Consolidation
        'keywords': ['bronchoscopy'],
        'label_name': 'Consolidation'
    },
    3: {  # Edema
        'keywords': ['lasix'],
        'label_name': 'Edema'
    },
    4: {  # Enlarged Cardiomediastinum
        'keywords': ['tachycardic'],
        'label_name': 'Enlarged Cardiomediastinum'
    },
    5: {  # Fracture
        'keywords': ['fractures'],
        'label_name': 'Fracture'
    },
    6: {  # Lung Lesion
        'keywords': ['metastatic'],
        'label_name': 'Lung Lesion'
    },
    7: {  # Lung Opacity
        'keywords': ['opacities'],
        'label_name': 'Lung Opacity'
    },
    8: {  # No Finding
        'keywords': ['lung'],
        'label_name': 'No Finding'
    },
    9: {  # Pleural Effusion
        'keywords': ['effusions'],
        'label_name': 'Pleural Effusion'
    },
    10: {  # Pleural Other
        'keywords': ['pleural'],
        'label_name': 'Pleural Other'
    },
    11: {  # Pneumonia
        'keywords': ['pneumonia'],
        'label_name': 'Pneumonia'
    },
    12: {  # Pneumothorax
        'keywords': ['pneumothorax'],
        'label_name': 'Pneumothorax'
    },
    13: {  # Support Devices
        'keywords': ['placement'],
        'label_name': 'Support Devices'
    }
}

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

def load_keywords_from_config(config_path=None):
    """
    Load keywords from configuration file
    If config_path is None, use default PATHOLOGY_KEYWORDS
    """
    if config_path is None:
        print("No config path provided, using default PATHOLOGY_KEYWORDS")
        return PATHOLOGY_KEYWORDS
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            keywords_config = json.load(f)
            print(f"Loaded keywords config from {config_path}")

        # Convert string keys to integers for consistency
        converted_config = {}
        for key, value in keywords_config.items():
            try:
                int_key = int(key)
                converted_config[int_key] = value
            except ValueError:
                # If key cannot be converted to int, keep as string
                converted_config[key] = value
        
        return converted_config
    except FileNotFoundError:
        print(f"Config file {config_path} not found, using default keywords")
        return PATHOLOGY_KEYWORDS
    except json.JSONDecodeError:
        print(f"Invalid JSON in {config_path}, using default keywords")
        return PATHOLOGY_KEYWORDS


def check_mention_multiple_keywords(text, keywords_list):
    """
    Check if any of the keywords in the list are mentioned in the text
    Uses substring matching (case-insensitive) - if keyword is contained anywhere in text, it matches
    
    Args:
        text (str): Text to search in
        keywords_list (list): List of keywords/phrases to search for
        
    Returns:
        bool: True if any keyword is found, False otherwise
        dict: Details about which keywords were found
    """
    text_lower = text.lower()
    found_keywords = []
    
    for keyword in keywords_list:
        # Use simple substring matching for all keywords
        if keyword.lower() in text_lower:
            found_keywords.append(keyword)
    
    return len(found_keywords) > 0, {
        'found_keywords': found_keywords,
        'total_found': len(found_keywords)
    }


def analyze_mentions_enhanced(dataset, label_index, keywords_config):
    """
    Enhanced mention analysis using multiple keywords per label
    
    Args:
        dataset: DataFrame with processed text
        label_index: Index of the label to analyze
        keywords_config: Dictionary containing keywords configuration
        
    Returns:
        tuple: (positive_indices, negative_indices, mention_details)
    """
    # print(f"Debug: Looking for label_index {label_index} (type: {type(label_index)})")
    # print(f"Debug: Available keys in config: {list(keywords_config.keys())}")
    # print(f"Debug: Key types: {[type(k) for k in keywords_config.keys()]}")
    
    if label_index not in keywords_config:
        raise ValueError(f"Label index {label_index} not found in keywords configuration. Available keys: {list(keywords_config.keys())}")
    
    keywords_list = keywords_config[label_index]['keywords']
    label_name = keywords_config[label_index]['label_name']
    
    print(f"Analyzing mentions for {label_name} using keywords: {keywords_list}")
    
    # Check mentions for each text
    mention_results = []
    for idx, text in enumerate(dataset['processed_text']):
        has_mention, details = check_mention_multiple_keywords(text, keywords_list)
        mention_results.append({
            'index': dataset.index[idx],
            'has_mention': has_mention,
            'details': details
        })
    
    # Separate indices
    positive_indices = np.array([r['index'] for r in mention_results if r['has_mention']])
    negative_indices = np.array([r['index'] for r in mention_results if not r['has_mention']])
    
    # Statistics
    total_samples = len(mention_results)
    positive_count = len(positive_indices)
    negative_count = len(negative_indices)
    
    print(f"Total samples: {total_samples}")
    print(f"Samples with mentions: {positive_count} ({positive_count/total_samples*100:.1f}%)")
    print(f"Samples without mentions: {negative_count} ({negative_count/total_samples*100:.1f}%)")
    
    return positive_indices, negative_indices, {
        'label_name': label_name,
        'keywords_used': keywords_list,
        'mention_results': mention_results,
        'statistics': {
            'total': total_samples,
            'with_mentions': positive_count,
            'without_mentions': negative_count
        }
    }


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


def stratified_bootstrap_resample(labels, random_state=None):
    """Perform stratified bootstrap resampling while maintaining class balance"""
    from sklearn.utils import resample
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


def bootstrap_prior_mention_comparison(
    label_index,
    positive_indices,
    negative_indices,
    all_outputs,
    all_labels,
    num_bootstraps=100000,
    bonferroni=True,
    n_tests=13
):
    """
    Bootstrap metrics for samples with/without prior mentions using global threshold.

    Args:
        label_index: Index of the label to evaluate
        positive_indices: Indices of samples with prior mentions
        negative_indices: Indices of samples without prior mentions
        all_outputs: Model predictions
        all_labels: True labels
        num_bootstraps: Number of bootstrap iterations
        bonferroni: Whether to apply Bonferroni correction
        n_tests: Number of tests for Bonferroni correction

    Returns:
        Dictionary with metrics for positive, negative, and difference
    """
    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

    positive_metrics = {
        'auroc': [],
        'auprc': [],
        'sens_at_spec95': [],
        'sens_at_global_spec95': [],
        'spec_at_global_spec95': []
    }
    negative_metrics = {
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

    current_labels = all_labels[:, label_index]
    current_outputs = all_outputs[:, label_index]

    for i in tqdm.tqdm(range(num_bootstraps), desc="Bootstrapping"):
        # Stratified resampling from entire dataset
        boot_indices = stratified_bootstrap_resample(current_labels, random_state=i)
        boot_labels = current_labels[boot_indices]
        boot_outputs = current_outputs[boot_indices]

        # Skip if no class diversity
        if len(np.unique(boot_labels)) < 2:
            continue

        # Compute global threshold from entire bootstrap sample
        try:
            fpr_global, tpr_global, thresholds_global = roc_curve(boot_labels, boot_outputs)
            target_fpr = 0.05
            idx_global = np.argmin(np.abs(fpr_global - target_fpr))
            global_threshold = thresholds_global[idx_global]
        except ValueError:
            continue

        # Find which bootstrap indices correspond to positive/negative groups
        boot_positive_mask = np.isin(boot_indices, positive_indices)
        boot_negative_mask = np.isin(boot_indices, negative_indices)

        # Skip if either group is empty or has no class diversity
        if not np.any(boot_positive_mask) or not np.any(boot_negative_mask):
            continue

        pos_labels = boot_labels[boot_positive_mask]
        pos_outputs = boot_outputs[boot_positive_mask]
        neg_labels = boot_labels[boot_negative_mask]
        neg_outputs = boot_outputs[boot_negative_mask]

        if len(np.unique(pos_labels)) < 2 or len(np.unique(neg_labels)) < 2:
            continue

        # Calculate metrics for positive group (with mentions)
        try:
            pos_auroc = roc_auc_score(pos_labels, pos_outputs)
            pos_auprc = average_precision_score(pos_labels, pos_outputs)

            # Subgroup-specific threshold
            fpr_pos, tpr_pos, _ = roc_curve(pos_labels, pos_outputs)
            idx_pos = np.argmin(np.abs(fpr_pos - target_fpr))
            pos_sens = tpr_pos[idx_pos]

            # Global threshold metrics
            pos_outputs_np = pos_outputs if isinstance(pos_outputs, np.ndarray) else pos_outputs.cpu().numpy()
            pos_labels_np = pos_labels if isinstance(pos_labels, np.ndarray) else pos_labels.cpu().numpy()
            y_pred_binary_pos = (pos_outputs_np >= global_threshold).astype(int)

            tp_pos = np.sum((y_pred_binary_pos == 1) & (pos_labels_np == 1))
            fn_pos = np.sum((y_pred_binary_pos == 0) & (pos_labels_np == 1))
            tn_pos = np.sum((y_pred_binary_pos == 0) & (pos_labels_np == 0))
            fp_pos = np.sum((y_pred_binary_pos == 1) & (pos_labels_np == 0))

            pos_sens_global = tp_pos / (tp_pos + fn_pos) if (tp_pos + fn_pos) > 0 else np.nan
            pos_spec_global = tn_pos / (tn_pos + fp_pos) if (tn_pos + fp_pos) > 0 else np.nan
        except ValueError:
            continue

        # Calculate metrics for negative group (without mentions)
        try:
            neg_auroc = roc_auc_score(neg_labels, neg_outputs)
            neg_auprc = average_precision_score(neg_labels, neg_outputs)

            # Subgroup-specific threshold
            fpr_neg, tpr_neg, _ = roc_curve(neg_labels, neg_outputs)
            idx_neg = np.argmin(np.abs(fpr_neg - target_fpr))
            neg_sens = tpr_neg[idx_neg]

            # Global threshold metrics
            neg_outputs_np = neg_outputs if isinstance(neg_outputs, np.ndarray) else neg_outputs.cpu().numpy()
            neg_labels_np = neg_labels if isinstance(neg_labels, np.ndarray) else neg_labels.cpu().numpy()
            y_pred_binary_neg = (neg_outputs_np >= global_threshold).astype(int)

            tp_neg = np.sum((y_pred_binary_neg == 1) & (neg_labels_np == 1))
            fn_neg = np.sum((y_pred_binary_neg == 0) & (neg_labels_np == 1))
            tn_neg = np.sum((y_pred_binary_neg == 0) & (neg_labels_np == 0))
            fp_neg = np.sum((y_pred_binary_neg == 1) & (neg_labels_np == 0))

            neg_sens_global = tp_neg / (tp_neg + fn_neg) if (tp_neg + fn_neg) > 0 else np.nan
            neg_spec_global = tn_neg / (tn_neg + fp_neg) if (tn_neg + fp_neg) > 0 else np.nan

            # Append metrics
            positive_metrics['auroc'].append(pos_auroc)
            positive_metrics['auprc'].append(pos_auprc)
            positive_metrics['sens_at_spec95'].append(pos_sens)
            positive_metrics['sens_at_global_spec95'].append(pos_sens_global)
            positive_metrics['spec_at_global_spec95'].append(pos_spec_global)

            negative_metrics['auroc'].append(neg_auroc)
            negative_metrics['auprc'].append(neg_auprc)
            negative_metrics['sens_at_spec95'].append(neg_sens)
            negative_metrics['sens_at_global_spec95'].append(neg_sens_global)
            negative_metrics['spec_at_global_spec95'].append(neg_spec_global)

            diffs['auroc'].append(pos_auroc - neg_auroc)
            diffs['auprc'].append(pos_auprc - neg_auprc)
            diffs['sens_at_spec95'].append(pos_sens - neg_sens)
            diffs['sens_at_global_spec95'].append(pos_sens_global - neg_sens_global)
            diffs['spec_at_global_spec95'].append(pos_spec_global - neg_spec_global)
        except ValueError:
            continue

    # Handle empty results
    if not positive_metrics['auroc']:
        print(f"Warning: No valid bootstrap samples")
        return {
            'positive': {
                'auroc': {'mean': np.nan, 'ci': [np.nan, np.nan]},
                'auprc': {'mean': np.nan, 'ci': [np.nan, np.nan]},
                'sens_at_spec95': {'mean': np.nan, 'ci': [np.nan, np.nan]},
                'sens_at_global_spec95': {'mean': np.nan, 'ci': [np.nan, np.nan]},
                'spec_at_global_spec95': {'mean': np.nan, 'ci': [np.nan, np.nan]}
            },
            'negative': {
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
            'n_bootstraps': 0
        }

    # Calculate summary statistics with Bonferroni correction
    lower_pct, upper_pct = compute_ci_percentiles(alpha=0.05, n_tests=n_tests, bonferroni=bonferroni)

    results = {
        'positive': {
            'auroc': {
                'mean': np.mean(positive_metrics['auroc']),
                'ci': np.percentile(positive_metrics['auroc'], [lower_pct, upper_pct])
            },
            'auprc': {
                'mean': np.mean(positive_metrics['auprc']),
                'ci': np.percentile(positive_metrics['auprc'], [lower_pct, upper_pct])
            },
            'sens_at_spec95': {
                'mean': np.mean(positive_metrics['sens_at_spec95']),
                'ci': np.percentile(positive_metrics['sens_at_spec95'], [lower_pct, upper_pct])
            },
            'sens_at_global_spec95': {
                'mean': np.mean(positive_metrics['sens_at_global_spec95']),
                'ci': np.percentile(positive_metrics['sens_at_global_spec95'], [lower_pct, upper_pct])
            },
            'spec_at_global_spec95': {
                'mean': np.mean(positive_metrics['spec_at_global_spec95']),
                'ci': np.percentile(positive_metrics['spec_at_global_spec95'], [lower_pct, upper_pct])
            }
        },
        'negative': {
            'auroc': {
                'mean': np.mean(negative_metrics['auroc']),
                'ci': np.percentile(negative_metrics['auroc'], [lower_pct, upper_pct])
            },
            'auprc': {
                'mean': np.mean(negative_metrics['auprc']),
                'ci': np.percentile(negative_metrics['auprc'], [lower_pct, upper_pct])
            },
            'sens_at_spec95': {
                'mean': np.mean(negative_metrics['sens_at_spec95']),
                'ci': np.percentile(negative_metrics['sens_at_spec95'], [lower_pct, upper_pct])
            },
            'sens_at_global_spec95': {
                'mean': np.mean(negative_metrics['sens_at_global_spec95']),
                'ci': np.percentile(negative_metrics['sens_at_global_spec95'], [lower_pct, upper_pct])
            },
            'spec_at_global_spec95': {
                'mean': np.mean(negative_metrics['spec_at_global_spec95']),
                'ci': np.percentile(negative_metrics['spec_at_global_spec95'], [lower_pct, upper_pct])
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
        'n_bootstraps': len(positive_metrics['auroc'])
    }

    return results


# Updated main function
def main(label_index, use_test_data=False, keywords_config_path=None,
         train_path=None, val_path=None, test_path=None,
         output_parent_dir=None, num_bootstraps=100000, bonferroni=True, n_tests=13):
    
    # Default data paths
    default_train_path = 'experiments/train.csv.gz'
    default_val_path = 'experiments/val.csv.gz'
    default_test_path = 'experiments/experiments.test.csv.gz'
    
    # Use provided paths or defaults
    train_path = train_path or default_train_path
    val_path = val_path or default_val_path
    test_path = test_path or default_test_path
    
    print(f"Loading datasets:")
    print(f"  Train: {train_path}")
    print(f"  Val: {val_path}")
    print(f"  Test: {test_path}")
    
    # Load datasets
    train_df = pd.read_csv(train_path, compression='gzip' if train_path.endswith('.gz') else None)
    val_df = pd.read_csv(val_path, compression='gzip' if val_path.endswith('.gz') else None)
    test_df = pd.read_csv(test_path, compression='gzip' if test_path.endswith('.gz') else None)

    label_columns = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
                    'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other', 
                    'Pneumonia', 'Pneumothorax', 'Support Devices']
    
    # Handle NA values (same as before)
    train_df[label_columns] = train_df[label_columns].fillna(0)
    val_df[label_columns] = val_df[label_columns].fillna(0)
    test_df[label_columns] = test_df[label_columns].fillna(0)

    # Select dataset and construct output directory
    if use_test_data:
        dataset = test_df
        print("Using TEST dataset for evaluation")
    else:
        dataset = pd.concat([train_df, val_df])
        print("Using TRAIN+VAL dataset for evaluation")
    
    # Construct output directory with prefix and suffix
    bonferroni_suffix = "_bonferroni" if bonferroni else ""
    if output_parent_dir:
        # Use provided prefix
        output_dir = output_parent_dir + ('_test' if use_test_data else '')
    else:
        # Use default prefix with dynamic bonferroni suffix
        default_prefix = f'prior_mention_evaluation_rbtl{bonferroni_suffix}'
        output_dir = default_prefix + ('_test' if use_test_data else '')

    print(f"Output directory: {output_dir}")
    print(f"Bootstrap iterations: {num_bootstraps}")
    
    # Load model outputs
    all_outputs = load_model_outputs(use_test_data)
    all_labels = torch.Tensor(dataset[label_columns].values).int()

    # Preprocess text
    dataset['processed_text'] = dataset['text'].map(chunk)
    
    # Load keywords configuration
    keywords_config = load_keywords_from_config(keywords_config_path)
    
    # Enhanced mention analysis
    positive_indices, negative_indices, mention_details = analyze_mentions_enhanced(
        dataset, label_index, keywords_config
    )

    # Bootstrap analysis with configurable iterations - using independent function
    correction_msg = f"with Bonferroni correction (n_tests={n_tests})" if bonferroni else "without Bonferroni correction"
    print(f"[INFO] Starting bootstrap analysis with {num_bootstraps} iterations {correction_msg}...")

    metrics_results = bootstrap_prior_mention_comparison(
        label_index=label_index,
        positive_indices=positive_indices,
        negative_indices=negative_indices,
        all_outputs=all_outputs,
        all_labels=all_labels,
        num_bootstraps=num_bootstraps,
        bonferroni=bonferroni,
        n_tests=n_tests
    )

    results = {
        "performance_on_previous_mentions": metrics_results['positive'],
        "performance_on_no_previous_mentions": metrics_results['negative'],
        "diffs": metrics_results['difference'],
        "dataset_used": "test" if use_test_data else "train_val",
        "mention_analysis": mention_details,
        "keywords_config_used": keywords_config_path or "default",
        "num_bootstraps_used": num_bootstraps,
        "output_directory": output_dir
    }
    
    # Print results summary before saving
    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)
    print(f"Label: {mention_details['label_name']}")
    print(f"Keywords used: {mention_details['keywords_used']}")
    print(f"Dataset: {'TEST' if use_test_data else 'TRAIN+VAL'}")
    print(f"Bootstrap iterations: {num_bootstraps}")
    print()
    print("Performance Metrics:")
    print(f"  Performance on samples WITH mentions:")
    print(f"    AUROC: {metrics_results['positive']['auroc']['mean']:.4f} (CI: {metrics_results['positive']['auroc']['ci'][0]:.4f}, {metrics_results['positive']['auroc']['ci'][1]:.4f})")
    print(f"    AUPRC: {metrics_results['positive']['auprc']['mean']:.4f} (CI: {metrics_results['positive']['auprc']['ci'][0]:.4f}, {metrics_results['positive']['auprc']['ci'][1]:.4f})")
    print(f"    Sensitivity @ 95% Spec (subgroup threshold): {metrics_results['positive']['sens_at_spec95']['mean']:.4f} (CI: {metrics_results['positive']['sens_at_spec95']['ci'][0]:.4f}, {metrics_results['positive']['sens_at_spec95']['ci'][1]:.4f})")
    print(f"    Sensitivity @ 95% Spec (global threshold): {metrics_results['positive']['sens_at_global_spec95']['mean']:.4f} (CI: {metrics_results['positive']['sens_at_global_spec95']['ci'][0]:.4f}, {metrics_results['positive']['sens_at_global_spec95']['ci'][1]:.4f})")
    print(f"    Specificity at global threshold: {metrics_results['positive']['spec_at_global_spec95']['mean']:.4f} (CI: {metrics_results['positive']['spec_at_global_spec95']['ci'][0]:.4f}, {metrics_results['positive']['spec_at_global_spec95']['ci'][1]:.4f})")
    print(f"  Performance on samples WITHOUT mentions:")
    print(f"    AUROC: {metrics_results['negative']['auroc']['mean']:.4f} (CI: {metrics_results['negative']['auroc']['ci'][0]:.4f}, {metrics_results['negative']['auroc']['ci'][1]:.4f})")
    print(f"    AUPRC: {metrics_results['negative']['auprc']['mean']:.4f} (CI: {metrics_results['negative']['auprc']['ci'][0]:.4f}, {metrics_results['negative']['auprc']['ci'][1]:.4f})")
    print(f"    Sensitivity @ 95% Spec (subgroup threshold): {metrics_results['negative']['sens_at_spec95']['mean']:.4f} (CI: {metrics_results['negative']['sens_at_spec95']['ci'][0]:.4f}, {metrics_results['negative']['sens_at_spec95']['ci'][1]:.4f})")
    print(f"    Sensitivity @ 95% Spec (global threshold): {metrics_results['negative']['sens_at_global_spec95']['mean']:.4f} (CI: {metrics_results['negative']['sens_at_global_spec95']['ci'][0]:.4f}, {metrics_results['negative']['sens_at_global_spec95']['ci'][1]:.4f})")
    print(f"    Specificity at global threshold: {metrics_results['negative']['spec_at_global_spec95']['mean']:.4f} (CI: {metrics_results['negative']['spec_at_global_spec95']['ci'][0]:.4f}, {metrics_results['negative']['spec_at_global_spec95']['ci'][1]:.4f})")
    print(f"  Difference (WITH - WITHOUT):")
    print(f"    AUROC Diff: {metrics_results['difference']['auroc']['mean']:.4f} (CI: {metrics_results['difference']['auroc']['ci'][0]:.4f}, {metrics_results['difference']['auroc']['ci'][1]:.4f})")
    print(f"    AUPRC Diff: {metrics_results['difference']['auprc']['mean']:.4f} (CI: {metrics_results['difference']['auprc']['ci'][0]:.4f}, {metrics_results['difference']['auprc']['ci'][1]:.4f})")
    print(f"    Sensitivity Diff (subgroup threshold): {metrics_results['difference']['sens_at_spec95']['mean']:.4f} (CI: {metrics_results['difference']['sens_at_spec95']['ci'][0]:.4f}, {metrics_results['difference']['sens_at_spec95']['ci'][1]:.4f})")
    print(f"    Sensitivity Diff (global threshold): {metrics_results['difference']['sens_at_global_spec95']['mean']:.4f} (CI: {metrics_results['difference']['sens_at_global_spec95']['ci'][0]:.4f}, {metrics_results['difference']['sens_at_global_spec95']['ci'][1]:.4f})")
    print(f"    Specificity Diff (global threshold): {metrics_results['difference']['spec_at_global_spec95']['mean']:.4f} (CI: {metrics_results['difference']['spec_at_global_spec95']['ci'][0]:.4f}, {metrics_results['difference']['spec_at_global_spec95']['ci'][1]:.4f})")
    print()
    print("Sample Distribution:")
    print(f"  Total samples: {mention_details['statistics']['total']}")
    print(f"  With mentions: {mention_details['statistics']['with_mentions']} ({mention_details['statistics']['with_mentions']/mention_details['statistics']['total']*100:.1f}%)")
    print(f"  Without mentions: {mention_details['statistics']['without_mentions']} ({mention_details['statistics']['without_mentions']/mention_details['statistics']['total']*100:.1f}%)")
    print("="*60)

    # Save results
    label_name = mention_details['label_name']
    os.makedirs(output_dir, exist_ok=True)
    output_file = f'{output_dir}/{label_name}_evaluation.pkl'
    
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results saved to {output_file}")


# Updated entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced evaluation with multiple keywords per label")
    parser.add_argument('--label_index', type=int, help='Index of the label', required=True)
    parser.add_argument('--use_test_data', action='store_true', help='Use test dataset instead of train+val')
    parser.add_argument('--keywords_config', type=str, help='Path to keywords configuration JSON file', default=None)
    
    # Data path arguments
    parser.add_argument('--train_path', type=str, 
                       help='Path to training data CSV file', 
                       default='experiments/train.csv.gz')
    parser.add_argument('--val_path', type=str, 
                       help='Path to validation data CSV file', 
                       default='experiments/val.csv.gz')
    parser.add_argument('--test_path', type=str, 
                       help='Path to test data CSV file', 
                       default='experiments/experiments.test.csv.gz')
    
    # Output and bootstrap arguments
    parser.add_argument('--output_dir', type=str,
                       help='Output directory prefix (suffix "_test" will be auto-added for test data)',
                       default=None)
    parser.add_argument('--num_bootstraps', type=int,
                       help='Number of bootstrap iterations (default: 100000)',
                       default=100000)
    parser.add_argument('--no-bonferroni', dest='bonferroni',
                       action='store_false', default=True,
                       help='Disable Bonferroni correction (default: enabled)')
    parser.add_argument('--n_tests', type=int, default=13,
                       help='Number of simultaneous tests for Bonferroni correction (default: 13)')

    args = parser.parse_args()

    main(args.label_index, args.use_test_data, args.keywords_config,
         args.train_path, args.val_path, args.test_path,
         args.output_dir, args.num_bootstraps, args.bonferroni, args.n_tests)