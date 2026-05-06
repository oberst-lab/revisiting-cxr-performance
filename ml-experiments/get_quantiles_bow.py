import os
import pickle
import numpy as np
import pandas as pd
import argparse
from sklearn.base import ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV

def load_vectorizer(vectorizer_path):
    """Loads a pickled scikit-learn vectorizer."""
    with open(vectorizer_path, 'rb') as f:
        return pickle.load(f)

def load_model(model_path):
    """Loads a pickled scikit-learn classifier."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def evaluate_bow_quantiles(model_path: str,
                           label_name: str,
                           labels: np.ndarray,
                           bow_features,
                           output_dir: str):
    """
    Evaluates a pre-trained BoW classifier and saves prediction indices 
    grouped by probability quantiles (bottom 25%, middle 50%, top 25%).
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
            calibrated_clf.fit(bow_features, labels)
            y_prob = calibrated_clf.predict_proba(bow_features)[:, 1]
        except Exception as e:
            raise RuntimeError(f"Probability calibration failed for {label_name}: {str(e)}")
    else:
        y_prob = clf.predict_proba(bow_features)[:, 1]
    
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
        'top_25': top_25_indices
    }
    
    # Save the dictionary of quantile indices
    with open(quantile_predictions_path, 'wb') as f:
        pickle.dump(quantile_predictions, f)
    
    print(f"[SUCCESS] Saved quantile predictions for {label_name} to {quantile_predictions_path}")
    print(f"  - Bottom 25%: {len(bottom_25_indices)} samples")
    print(f"  - Middle 50%: {len(middle_50_indices)} samples")
    print(f"  - Top 25%:    {len(top_25_indices)} samples")
    
    return quantile_predictions

def get_best_classifiers_from_metrics(metrics_path):
    """Load and process metrics to find best classifiers."""
    print("[INFO] Loading and consolidating metrics...")
    metrics_files = [f for f in os.listdir(metrics_path) if f.startswith("metrics_label_")]
    if not metrics_files:
        print(f"[ERROR] No metrics files found in {metrics_path}. Exiting.")
        return None
        
    metrics_df = pd.concat([pd.read_csv(os.path.join(metrics_path, f)) for f in metrics_files], ignore_index=True)
    best_df = metrics_df.sort_values(by='AUROC', ascending=False).drop_duplicates('label_name').copy()
    return best_df

def find_best_model_files(model_dir, label_columns):
    """Find the best model files for each label in the model directory."""
    best_models = {}
    
    for label_name in label_columns:
        # Look for model files that match this label
        model_files = [f for f in os.listdir(model_dir) 
                      if f.startswith(f"{label_name}_") and f.endswith("_bow.pkl")]
        
        if len(model_files) == 1:
            # Only one model file found, use it
            model_file = model_files[0]
            classifier_name = model_file.replace(f"{label_name}_", "").replace("_bow.pkl", "")
            best_models[label_name] = {
                'file': model_file,
                'classifier': classifier_name
            }
        elif len(model_files) > 1:
            print(f"[WARNING] Multiple model files found for {label_name}: {model_files}")
            print(f"[WARNING] Using the first one: {model_files[0]}")
            model_file = model_files[0]
            classifier_name = model_file.replace(f"{label_name}_", "").replace("_bow.pkl", "")
            best_models[label_name] = {
                'file': model_file,
                'classifier': classifier_name
            }
        else:
            print(f"[ERROR] No model file found for {label_name}")
    
    return best_models

def main():
    parser = argparse.ArgumentParser(description='Evaluate BoW classifiers and generate quantile predictions')
    parser.add_argument('--skip-metrics', action='store_true', 
                       help='Skip metrics loading and use model files directly from model directory')
    parser.add_argument('--metrics-path', type=str, default='bow_sweep_results/metrics',
                       help='Path to metrics directory')
    parser.add_argument('--model-dir', type=str, default='bow_sweep_results/models',
                       help='Path to models directory') 
    parser.add_argument('--output-dir', type=str, default='bow_quantile_predictions_test',
                       help='Output directory for quantile predictions')
    parser.add_argument('--test-data-path', type=str, default='experiments/experiments.test.csv.gz',
                       help='Path to test data CSV file')
    
    args = parser.parse_args()
    
    # --- Configuration ---
    os.makedirs(args.output_dir, exist_ok=True)

    label_columns = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum',
                     'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
                     'Pneumonia', 'Pneumothorax', 'Support Devices']

    # --- Load Best Classifiers Info ---
    if args.skip_metrics:
        print("[INFO] Skipping metrics loading, scanning model directory for best models...")
        best_models = find_best_model_files(args.model_dir, label_columns)
        
        # Create summary without AUROC scores
        best_df_summary = pd.DataFrame({
            'Label': [label for label in label_columns if label in best_models],
            'Classifier': [best_models[label]['classifier'] for label in label_columns if label in best_models],
            'AUROC': ['N/A'] * len([label for label in label_columns if label in best_models])
        })
    else:
        best_df = get_best_classifiers_from_metrics(args.metrics_path)
        if best_df is None:
            return
        
        # Create mapping of label to best classifier name
        best_classifiers = dict(zip(best_df['label_name'], best_df['classifier_name']))
        best_models = {label: {'classifier': best_classifiers[label]} 
                      for label in label_columns if label in best_classifiers}
        
        # Prepare summary
        best_df_summary = best_df[['label_name', 'classifier_name', 'AUROC']].rename(
            columns={'label_name': 'Label', 'classifier_name': 'Classifier'}
        )
        best_df_summary['AUROC'] = best_df_summary['AUROC'].apply(lambda x: float(f"{x:.3g}"))
    
    # --- Load and Process Data ---
    print("[INFO] Loading and preparing test data...")
    test_df = pd.read_csv(args.test_data_path, compression='gzip')
    test_df[label_columns] = test_df[label_columns].fillna(0)
    test_df["processed_text"] = test_df["text"].fillna("").apply(lambda x: " ".join(x.split("eotextdelimiter")))
    
    vectorizer = load_vectorizer(os.path.join(args.model_dir, "bow_vectorizer.pkl"))
    print("[INFO] Transforming test data with BoW vectorizer...")
    test_bow = vectorizer.transform(test_df["processed_text"])

    # --- Run Evaluation ---
    print("\n[INFO] Starting quantile evaluation for each label...")
    for label_name in label_columns:
        if label_name not in best_models:
            print(f"[WARNING] No best model found for {label_name}. Skipping.")
            continue
            
        print(f"--- Processing: {label_name} ---")
        
        if args.skip_metrics:
            model_path = os.path.join(args.model_dir, best_models[label_name]['file'])
        else:
            best_clf_name = best_models[label_name]['classifier']
            model_path = os.path.join(args.model_dir, f"{label_name}_{best_clf_name}_bow.pkl")
        
        if not os.path.exists(model_path):
            print(f"[ERROR] Model file not found: {model_path}. Skipping {label_name}.")
            continue
            
        test_labels = test_df[label_name].values
        evaluate_bow_quantiles(
            model_path=model_path,
            label_name=label_name,
            labels=test_labels,
            bow_features=test_bow,
            output_dir=args.output_dir
        )

    # --- Save Summary ---
    summary_path = os.path.join(args.output_dir, "best_bow_classifiers_summary.csv")
    best_df_summary.to_csv(summary_path, index=False)
    print(f"\n[SUCCESS] Best BoW classifier summary saved to {summary_path}")

if __name__ == "__main__":
    main()