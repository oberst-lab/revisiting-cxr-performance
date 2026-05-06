import os
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.base import ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV

def load_model(model_path):
    """Loads a pickled scikit-learn classifier."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def evaluate_text_quantiles(model_path: str,
                           label_name: str,
                           labels: np.ndarray,
                           embeddings: np.ndarray,
                           output_dir: str):
    """
    Evaluates a pre-trained text classifier and saves prediction indices 
    grouped by probability quantiles (bottom 25%, middle 50%, top 25%).

    Args:
        model_path (str): Path to the pickled classifier model.
        label_name (str): The name of the label being evaluated.
        labels (np.ndarray): The true labels for the dataset.
        embeddings: The text embeddings for the dataset.
        output_dir (str): The directory to save the quantile predictions.
    """
    # Ensure the specific output directory for the label exists
    label_output_dir = os.path.join(output_dir, label_name)
    os.makedirs(label_output_dir, exist_ok=True)
    quantile_predictions_path = os.path.join(label_output_dir, f'{label_name}_quantile_predictions.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    clf = load_model(model_path)
    if not isinstance(clf, ClassifierMixin):
        raise ValueError(f"Loaded object from {model_path} is not a scikit-learn classifier")
    
    # Get prediction probabilities, calibrating if necessary
    if not hasattr(clf, 'predict_proba'):
        print(f"[INFO] Calibrating classifier for {label_name} as predict_proba is not available.")
        calibrated_clf = CalibratedClassifierCV(clf, method='sigmoid', cv='prefit')
        try:
            # Fitting on the entire dataset since this is for evaluation, not training
            calibrated_clf.fit(embeddings, labels)
            y_prob = calibrated_clf.predict_proba(embeddings)[:, 1]
        except Exception as e:
            raise RuntimeError(f"Probability calibration failed for {label_name}: {str(e)}")
    else:
        y_prob = clf.predict_proba(embeddings)[:, 1]
    
    # Calculate quantile thresholds
    q25 = np.quantile(y_prob, 0.25)
    q75 = np.quantile(y_prob, 0.75)
    
    # Find indices for each quantile group
    bottom_25_indices = np.where(y_prob <= q25)[0]
    middle_50_indices = np.where((y_prob > q25) & (y_prob < q75))[0]
    top_25_indices = np.where(y_prob >= q75)[0]
    
    quantile_predictions = {
        'bottom_25': bottom_25_indices,
        'middle_50': middle_50_indices,
        'top_25': top_25_indices,
        'probabilities': y_prob  # Save the actual probabilities for reference
    }
    
    # Save the dictionary of quantile indices
    with open(quantile_predictions_path, 'wb') as f:
        pickle.dump(quantile_predictions, f)
    
    print(f"[SUCCESS] Saved quantile predictions for {label_name} to {quantile_predictions_path}")
    print(f"  - Bottom 25%: {len(bottom_25_indices)} samples (prob ≤ {q25:.3f})")
    print(f"  - Middle 50%: {len(middle_50_indices)} samples ({q25:.3f} < prob < {q75:.3f})")
    print(f"  - Top 25%:    {len(top_25_indices)} samples (prob ≥ {q75:.3f})")
    
    return quantile_predictions

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate text classifiers and generate quantile predictions')
    
    parser.add_argument('--metrics_df', type=str, required=True,
                        help='Path to the consolidated metrics CSV file')
    parser.add_argument('--test_embeddings', type=str, required=True,
                        help='Path to the test embeddings .pt file')
    parser.add_argument('--test_df', type=str, required=True,
                        help='Path to the test dataframe CSV file')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Directory containing the trained model files')
    parser.add_argument('--model_name_suffix', type=str, default='_train_val_split',
                        help='Suffix for model file names (default: _train_val_split)')
    parser.add_argument('--output_dir', type=str, default='quantile_predictions_test',
                        help='Output directory for quantile predictions (default: quantile_predictions_test)')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # --- Configuration from arguments ---
    metrics_file = args.metrics_df
    model_dir = args.model_path
    output_dir = args.output_dir
    embeddings_path = args.test_embeddings
    test_df_path = args.test_df
    model_suffix = args.model_name_suffix
    
    os.makedirs(output_dir, exist_ok=True)

    # --- Load and Process Data ---
    print("[INFO] Loading and consolidating metrics...")
    
    # Load the consolidated metrics file
    if not os.path.exists(metrics_file):
        print(f"[ERROR] Metrics file not found at {metrics_file}. Exiting.")
        return
        
    metrics_df = pd.read_csv(metrics_file)
    best_df = metrics_df.sort_values(by='AUROC', ascending=False).drop_duplicates('label_name').copy()

    label_columns = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum',
                     'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
                     'Pneumonia', 'Pneumothorax', 'Support Devices']
    
    print("[INFO] Loading and preparing test data...")
    if not os.path.exists(test_df_path):
        print(f"[ERROR] Test dataframe not found at {test_df_path}. Exiting.")
        return
        
    # Handle both compressed and uncompressed CSV files
    if test_df_path.endswith('.gz'):
        test_df = pd.read_csv(test_df_path, compression='gzip')
    else:
        test_df = pd.read_csv(test_df_path)
    test_df[label_columns] = test_df[label_columns].fillna(0)
    
    print("[INFO] Loading text embeddings...")
    if not os.path.exists(embeddings_path):
        print(f"[ERROR] Embeddings file not found at {embeddings_path}. Exiting.")
        return
        
    embeddings = torch.load(embeddings_path, weights_only=False)
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.numpy()

    # --- Run Evaluation ---
    print("\n[INFO] Starting quantile evaluation for each label...")
    for label_name in label_columns:
        if label_name not in best_df['label_name'].values:
            print(f"[WARNING] No best model found for {label_name}. Skipping.")
            continue
            
        print(f"--- Processing: {label_name} ---")
        best_clf_name = best_df.loc[best_df['label_name'] == label_name, 'classifier_name'].values[0]
        model_path = os.path.join(model_dir, f"{label_name}_{best_clf_name}{model_suffix}.pkl")
        
        test_labels = test_df[label_name].values
        
        try:
            evaluate_text_quantiles(
                model_path=model_path,
                label_name=label_name,
                labels=test_labels,
                embeddings=embeddings,
                output_dir=output_dir
            )
        except Exception as e:
            print(f"[ERROR] Failed to process {label_name}: {str(e)}")
            continue

    # --- Save Summary ---
    best_df_summary = best_df[['label_name', 'classifier_name', 'AUROC']].rename(
        columns={'label_name': 'Label', 'classifier_name': 'Classifier'}
    )
    best_df_summary['AUROC'] = best_df_summary['AUROC'].apply(lambda x: float(f"{x:.3g}"))
    summary_path = os.path.join(output_dir, "best_text_classifiers_summary.csv")
    best_df_summary.to_csv(summary_path, index=False)
    print(f"\n[SUCCESS] Best text classifier summary saved to {summary_path}")

if __name__ == "__main__":
    main()