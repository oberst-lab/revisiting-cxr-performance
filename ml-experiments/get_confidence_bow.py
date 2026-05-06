import os
import pickle
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from scipy.sparse import load_npz
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import ClassifierMixin

def load_vectorizer(vectorizer_path):
    with open(vectorizer_path, 'rb') as f:
        return pickle.load(f)

def load_model(model_path):
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def evaluate_bow_classifier(model_path: str,
                            label_name: str,
                            labels: np.ndarray,
                            bow_features,
                            output_dir: str,
                            mode='test'):
    """
    Evaluates a pre-trained BoW classifier on test data.
    Mimics the encoding_utils.evaluate_classifier() structure.
    Saves confident prediction indices for various thresholds.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('bow_confident_predictions_test', exist_ok=True)
    confident_predictions_path = f'bow_confident_predictions_test/{label_name}.pkl'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)
    if not isinstance(clf, ClassifierMixin):
        raise ValueError(f"Loaded object from {model_path} is not a scikit-learn classifier")
    # Get probabilities
    if not hasattr(clf, 'predict_proba'):
        calibrated_clf = CalibratedClassifierCV(clf, method='sigmoid', cv='prefit')
        try:
            calibrated_clf.fit(bow_features, labels)
            y_prob = calibrated_clf.predict_proba(bow_features)[:, 1]
        except Exception as e:
            raise RuntimeError(f"Probability calibration failed for {label_name}: {str(e)}")
    else:
        y_prob = clf.predict_proba(bow_features)[:, 1]
    confident_indices_dict = {}
    # Confidence levels 0.01–0.05 in 0.01 steps
    for confidence_level in np.arange(0.01, 0.06, 0.01):
        confident_indices = np.where((y_prob < confidence_level) | (y_prob > (1 - confidence_level)))[0]
        confident_indices_dict[confidence_level] = confident_indices
        print(f"[INFO] {label_name} | Confidence {confidence_level:.2f} : {len(confident_indices)} confident")
    # Confidence levels 0.05–0.45 in 0.05 steps
    for confidence_level in np.arange(0.05, 0.50, 0.05):
        confident_indices = np.where((y_prob < confidence_level) | (y_prob > (1 - confidence_level)))[0]
        confident_indices_dict[confidence_level] = confident_indices
        print(f"[INFO] {label_name} | Confidence {confidence_level:.2f} : {len(confident_indices)} confident")
    with open(confident_predictions_path, 'wb') as f:
        pickle.dump(confident_indices_dict, f)
    print(f"[SUCCESS] Saved confident predictions for {label_name} to {confident_predictions_path}")
    return confident_indices_dict

def main():
    metrics_path = "bow_sweep_results/metrics"
    model_dir = "bow_sweep_results/models"
    output_dir = "bow_confident_predictions_test"
    os.makedirs(output_dir, exist_ok=True)

    metrics_files = [f for f in os.listdir(metrics_path) if f.startswith("metrics_label_")]
    metrics_df = pd.concat([pd.read_csv(os.path.join(metrics_path, f)) for f in metrics_files], ignore_index=True)
    best_df = metrics_df.sort_values(by='AUROC', ascending=False).drop_duplicates('label_name').copy()

    label_columns = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum',
                     'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
                     'Pneumonia', 'Pneumothorax', 'Support Devices']

    test_df = pd.read_csv("experiments/experiments.test.csv.gz", compression='gzip')
    test_df[label_columns] = test_df[label_columns].fillna(0)
    test_df["processed_text"] = test_df["text"].fillna("").apply(lambda x: " ".join(x.split("eotextdelimiter")))

    with open(os.path.join(model_dir, "bow_vectorizer.pkl"), 'rb') as f:
        vectorizer = pickle.load(f)

    print("[INFO] Transforming test data...")
    test_bow = vectorizer.transform(test_df["processed_text"])

    for label_name in label_columns:
        if label_name not in best_df['label_name'].values:
            print(f"[WARNING] No best model for {label_name}. Skipping.")
            continue
        best_clf_name = best_df.loc[best_df['label_name'] == label_name, 'classifier_name'].values[0]
        model_path = os.path.join(model_dir, f"{label_name}_{best_clf_name}_bow.pkl")
        test_labels = test_df[label_name].values
        evaluate_bow_classifier(
            model_path=model_path,
            label_name=label_name,
            labels=test_labels,
            bow_features=test_bow,
            output_dir=output_dir,
            mode='test'
        )

    best_df_summary = best_df[['label_name', 'classifier_name', 'AUROC']].rename(
        columns={'label_name': 'Label', 'classifier_name': 'Classifier'}
    )
    best_df_summary['AUROC'] = best_df_summary['AUROC'].apply(lambda x: float(f"{x:.3g}"))
    best_df_summary.to_csv(os.path.join(output_dir, "best_bow_classifiers.csv"), index=False)
    print(f"[SUCCESS] Best BoW classifier summary saved to {output_dir}/best_bow_classifiers.csv")

if __name__ == "__main__":
    main()
