import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
from scipy.optimize import linear_sum_assignment
import pickle
import os
from tqdm import tqdm
import argparse
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

def nearest_neighbor_matching(data_df, label_column, text_probs):
    """
    Perform optimal 1:1 nearest-neighbor matching using the Hungarian algorithm.
    
    Args:
        data_df: DataFrame containing the data and labels
        label_column: Name of the column with binary labels (1=positive, 0=negative)
        text_probs: Array of predicted probabilities from text classifier
        
    Returns:
        Dictionary with matched pairs of indices:
        {
            "positive": array of positive indices,
            "negative": array of matched negative indices
        }
    """
    # Get indices of positive and negative cases
    pos_indices = data_df[data_df[label_column] == 1].index.values
    neg_indices = data_df[data_df[label_column] == 0].index.values
    # Return empty if no pairs possible
    if len(pos_indices) == 0 or len(neg_indices) == 0:
        return {"positive": [], "negative": []}
    # Prepare probability arrays
    pos_probs = text_probs[pos_indices].reshape(-1, 1)
    neg_probs = text_probs[neg_indices].reshape(-1, 1)
    # Create cost matrix (absolute difference between probabilities)
    cost_matrix = np.abs(pos_probs - neg_probs.T)
    # Solve optimal assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # Handle case where we have more positives than negatives
    if len(pos_indices) > len(neg_indices):
        row_ind = row_ind[:len(neg_indices)]
        col_ind = col_ind[:len(neg_indices)]
    # Return matched pairs (original indices)
    matched_pairs = {
        "positive": pos_indices[row_ind],
        "negative": neg_indices[col_ind]
    }
    return matched_pairs


def bootstrap_matching(
    data_df, label_column, text_probs, vision_labels, vision_probs, n_bootstraps=100000
):
    """Bootstrap the entire matching process and calculate AUROC for each iteration."""
    aurocs = []
    n_matches = []
    data_df = data_df.reset_index(drop=True)

    for _ in tqdm(
        range(n_bootstraps), desc=f"Bootstrapping {label_column}", leave=False
    ):
        boot_df = data_df.sample(frac=1, replace=True)
        boot_text_probs = text_probs[boot_df.index]
        boot_vision_probs = vision_probs[boot_df.index]
        boot_vision_labels = vision_labels[boot_df.index]

        matched_indices = nearest_neighbor_matching(
            boot_df, label_column, boot_text_probs
        )

        if (
            matched_indices["positive"] is not None
            and len(matched_indices["positive"]) > 0
        ):
            true_labels = np.concatenate(
                [
                    boot_vision_labels[matched_indices["positive"]],
                    boot_vision_labels[matched_indices["negative"]],
                ]
            )
            pred_probs = np.concatenate(
                [
                    boot_vision_probs[matched_indices["positive"]],
                    boot_vision_probs[matched_indices["negative"]],
                ]
            )
            
            # Skip if only one class is present
            if len(np.unique(true_labels)) == 1:
                continue
                
            try:
                auroc = roc_auc_score(true_labels, pred_probs)
                aurocs.append(auroc)
                n_matches.append(len(matched_indices["positive"]))
            except ValueError:
                continue

    if aurocs:
        return {
            "auroc_mean": np.mean(aurocs),
            "auroc_ci": np.percentile(aurocs, [2.5, 97.5]),
            "avg_matches": np.mean(n_matches),
            "n_bootstraps": len(aurocs),
        }
    return {
        "auroc_mean": np.nan,
        "auroc_ci": [np.nan, np.nan],
        "avg_matches": 0,
        "n_bootstraps": 0,
    }

def main(label_index):
    """Run experiment for a single label specified by index."""
    n_bootstraps = 100000
    test_df = pd.read_csv(
        "experiments/experiments.test.csv.gz", compression="gzip"
    )
    label_columns = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Enlarged Cardiomediastinum",
        "Fracture",
        "Lung Lesion",
        "No Finding",
        "Lung Opacity",
        "Pleural Effusion",
        "Pleural Other",
        "Pneumonia",
        "Pneumothorax",
        "Support Devices",
    ]
    test_df[label_columns] = test_df[label_columns].fillna(0)
    embeddings = torch.load("experiments/test.pt", weights_only=False)

    label_name = label_columns[label_index]
    vision_labels = test_df[label_name].values
    metrics_df = pd.read_csv("text_classifiers_metrics_full_train/all_results.csv")
    best_df = metrics_df.sort_values(by="AUROC", ascending=False).drop_duplicates(
        "label_name"
    )
    clf_name = str(
        best_df[best_df["label_name"] == label_name]["classifier_name"].values[0]
    )
    model_path = f"text_models/{label_name}_{clf_name}_full_train.pkl"

    with open(model_path, "rb") as f:
        clf = pickle.load(f)

    # Handle classifiers that don't support predict_proba natively
    if not hasattr(clf, "predict_proba"):
        calibrated_clf = CalibratedClassifierCV(clf, method="sigmoid", cv="prefit")
        try:
            calibrated_clf.fit(embeddings, vision_labels) 
            y_prob = calibrated_clf.predict_proba(embeddings)[:, 1]
        except Exception as e:
            raise RuntimeError(f"Probability calibration failed: {str(e)}")
    else:
        y_prob = clf.predict_proba(embeddings)[:, 1]

    predictions_dir = "vision_predictions_test" 
    with open(f'{predictions_dir}/full_predictions.pkl', 'rb') as f:
        all_outputs = pickle.load(f)

    print(f"\nProcessing {label_name} (index {label_index})...")

    # Run bootstrap matching on the full dataset (no confidence splitting)
    results = bootstrap_matching(
        data_df=test_df,
        label_column=label_name,
        text_probs=y_prob,
        vision_labels=vision_labels,
        vision_probs = all_outputs[:, label_index],
        n_bootstraps=n_bootstraps,
    )

    # Save results
    os.makedirs("matching_neighbors", exist_ok=True)
    output_file = f"matching_neighbors/{label_name}_results.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(results, f)

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run bootstrap matching for a specific label."
    )
    parser.add_argument(
        "--label_index", type=int, required=True, help="Index of the label to process"
    )
    args = parser.parse_args()

    main(label_index=args.label_index)
