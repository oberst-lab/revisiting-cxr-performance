import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import vstack as sparse_vstack
from sklearn import metrics
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron, RidgeClassifierCV, PassiveAggressiveClassifier, SGDClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from inspect import signature
import sklearn

sklearn.set_config(enable_metadata_routing=True)

CLASSIFIERS = [
    ("Perceptron", Perceptron(penalty='l2', random_state=42, max_iter=1000, tol=1e-3)),
    ("RidgeClassifierCV", RidgeClassifierCV(alphas=np.logspace(-3, 3, 7))),
    ("PassiveAggressiveClassifier", PassiveAggressiveClassifier(random_state=42, max_iter=1000, tol=1e-3)),
    ("RandomForestClassifier", RandomForestClassifier(random_state=42)),
    ("LinearSVC", LinearSVC(max_iter=2000, random_state=42, dual='auto')),
    ("SGDClassifier", SGDClassifier(loss='log_loss', penalty='l2', random_state=42, max_iter=1000, tol=1e-3)),
    ("DecisionTreeClassifier", DecisionTreeClassifier(random_state=42))
]

PARAM_GRIDS = {
    "Perceptron": {'alpha': [1e-6, 1e-5, 1e-4]},
    "RidgeClassifierCV": {'alphas': [np.logspace(-4, 4, 9)]},
    "PassiveAggressiveClassifier": {'C': [0.001, 0.01, 0.1, 1.0]},
    "RandomForestClassifier": {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None], 'min_samples_leaf': [1, 5, 10]},
    "LinearSVC": {'C': [0.01, 0.1, 1.0, 10.0]},
    "SGDClassifier": {'alpha': [1e-6, 1e-5, 1e-4], 'penalty': ['l1', 'l2', 'elasticnet']},
    "DecisionTreeClassifier": {'max_depth': [5, 10, 20, None], 'min_samples_leaf': [1, 5, 10, 20]}
}

def chunk(note: str) -> str:
    if not isinstance(note, str):
        return ""
    split_note = note.split("eotextdelimiter")
    return " ".join(split_note)

def train_and_evaluate(classifier_index, label_name, train_df, val_df,
                         train_bow, val_bow, output_dir, perform_hypersearch=False,
                         vectorizer=None):

    clf_name, base_clf = CLASSIFIERS[classifier_index]
    print(f"\n--- Processing: {clf_name} for label '{label_name}' ---")

    model_dir = os.path.join(output_dir, 'models')
    metrics_dir = os.path.join(output_dir, 'metrics')
    plots_dir = os.path.join(output_dir, 'plots')
    feature_importance_dir = os.path.join(output_dir, 'feature_importances')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(feature_importance_dir, exist_ok=True)

    model_path = os.path.join(model_dir, f'{label_name}_{clf_name}_bow.pkl')
    calibration_plot_path = os.path.join(plots_dir, f'calibration_plot_{label_name}_{clf_name}_bow.pdf')
    metrics_csv_path = os.path.join(metrics_dir, f'metrics_label_{label_name}_{clf_name}_bow.csv')
    feature_importance_path = os.path.join(feature_importance_dir, f'feature_importances_{label_name}_{clf_name}_bow.csv')

    clf = None
    if os.path.exists(model_path):
        print(f"[INFO] Found existing model at {model_path}. Loading it for evaluation and feature importance.")
        with open(model_path, 'rb') as f:
            clf = pickle.load(f)
        # Skip training/calibration steps as model is loaded
    else:
        print(f"[INFO] No existing model found at {model_path}. Proceeding with training and calibration.")
        print("[INFO] Training model on training set only...")

        train_labels = train_df[label_name].values
        train_groups = train_df["subject_id"].values
        train_features = train_bow
        val_features = val_bow

        if isinstance(base_clf, (GaussianNB, LinearDiscriminantAnalysis)):
            print(f"[WARNING] Converting BoW to dense array for {clf_name}. This may require significant memory.")
            try:
                train_features = train_features.toarray()
                val_features = val_features.toarray()
                print("[INFO] Conversion to dense array successful.")
            except MemoryError:
                print(f"[ERROR] MemoryError: Cannot convert sparse BoW to dense for {clf_name}. Skipping this classifier.")
                metrics_data = {'label_name': label_name, 'classifier_name': clf_name, 'AUROC': np.nan, 'Error': 'MemoryError during dense conversion'}
                pd.DataFrame([metrics_data]).to_csv(metrics_csv_path, index=False)
                return

        best_clf = base_clf
        grid_search_results = None

        if perform_hypersearch and clf_name in PARAM_GRIDS and PARAM_GRIDS[clf_name]:
            print(f"[INFO] Performing hyperparameter search (5-fold StratifiedGroupKFold) for {clf_name}...")
            param_grid = PARAM_GRIDS[clf_name]
            group_kfold = StratifiedGroupKFold(n_splits=5)
            grid_search = GridSearchCV(base_clf, param_grid, cv=group_kfold, scoring='roc_auc', n_jobs=-1, verbose=1, error_score='raise')

            try:
                grid_search.fit(train_features, train_labels, groups=train_groups)
                best_clf = grid_search.best_estimator_
                grid_search_results = grid_search
                print(f"[INFO] Best hyperparameters for {clf_name}: {grid_search.best_params_}")
                print(f"[INFO] Best CV AUROC score for {clf_name}: {grid_search.best_score_:.4f}")
            except Exception as e:
                print(f"[ERROR] GridSearchCV failed for {clf_name}: {e}. Using default parameters.")

        else:
            print(f"[INFO] Skipping hyperparameter search for {clf_name} (not requested or no grid defined). Using default parameters.")

        print(f"[INFO] Calibrating the best model for {clf_name} on training set using 5-fold StratifiedGroupKFold...")
        calibration_cv = StratifiedGroupKFold(n_splits=5)
        calibration_method = 'isotonic'
        if not (hasattr(best_clf, "predict_proba") or hasattr(best_clf, "decision_function")):
            print(f"[WARNING] {clf_name} does not support predict_proba or decision_function. Skipping calibration.")
            clf = best_clf
        else:
            clf = CalibratedClassifierCV(estimator=best_clf, cv=calibration_cv, method=calibration_method)
            try:
                clf.fit(train_features, train_labels, groups=train_groups)
            except Exception as e:
                print(f"[ERROR] Calibration failed for {clf_name}: {e}. Using uncalibrated model.")
                clf = best_clf
                if not hasattr(clf, 'classes_'):
                    try:
                        clf.fit(train_features, train_labels)
                    except Exception as fit_e:
                        print(f"[ERROR] Fitting uncalibrated {clf_name} failed after calibration error: {fit_e}. Skipping.")
                        metrics_data = {'label_name': label_name, 'classifier_name': clf_name, 'AUROC': np.nan, 'Error': f'Calibration and Fit failed: {e}; {fit_e}'}
                        pd.DataFrame([metrics_data]).to_csv(metrics_csv_path, index=False)
                        return

        print(f"[INFO] Saving final model to {model_path}")
        with open(model_path, 'wb') as f:
            pickle.dump(clf, f)

    # Ensure clf is defined before proceeding to evaluation and feature importance
    if clf is None:
        print(f"[ERROR] Model for {clf_name} could not be loaded or trained. Skipping evaluation and feature importance.")
        return

    print(f"[INFO] Evaluating {clf_name} on the validation set...")
    val_labels = val_df[label_name].values
    auroc = np.nan
    prob_true, prob_pred = np.array([]), np.array([])

    # Need to handle dense conversion for validation set if needed
    val_features = val_bow
    if isinstance(base_clf, (GaussianNB, LinearDiscriminantAnalysis)):
        try:
            val_features = val_features.toarray()
        except MemoryError:
            print(f"[ERROR] MemoryError: Cannot convert validation BoW to dense for {clf_name}. Skipping evaluation.")
            metrics_data = {'label_name': label_name, 'classifier_name': clf_name, 'AUROC': np.nan, 'Error': 'MemoryError during validation dense conversion'}
            pd.DataFrame([metrics_data]).to_csv(metrics_csv_path, index=False)
            return

    try:
        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(val_features)[:, 1]
        elif hasattr(clf, "decision_function"):
            y_scores = clf.decision_function(val_features)
            if y_scores.ndim > 1 and y_scores.shape[1] > 1:
                positive_class_idx = np.where(clf.classes_ == 1)[0][0] if 1 in clf.classes_ else 0
                y_scores = y_scores[:, positive_class_idx]

            y_prob = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min() + 1e-8)
        else:
            y_pred = clf.predict(val_features)
            y_prob = y_pred
            print(f"[WARNING] {clf_name} does not provide probabilities or scores. Using predictions for evaluation.")
            if len(np.unique(val_labels)) == 2:
                auroc = metrics.accuracy_score(val_labels, y_pred)
                print(f"[INFO] Reporting Accuracy instead of AUROC for {clf_name}: {auroc:.4f}")

        if len(np.unique(val_labels)) == 2 and (hasattr(clf, "predict_proba") or hasattr(clf, "decision_function")):
            auroc = metrics.roc_auc_score(val_labels, y_prob)
            print(f"[INFO] Validation Set AUROC for {clf_name}: {auroc:.4f}")
            prob_true, prob_pred = calibration_curve(val_labels, y_prob, n_bins=10, strategy='uniform')
        elif auroc is np.nan:
            pass
        else:
            print(f"[INFO] Skipping AUROC calculation for {clf_name} due to lack of probabilities/scores or non-binary labels.")

    except Exception as e:
        print(f"[ERROR] Evaluation failed for {clf_name} on validation set: {e}")

    if prob_pred.size > 0 and prob_true.size > 0:
        plt.figure(figsize=(8, 8))
        plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label=f'{clf_name} (AUROC: {auroc:.3f})')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
        plt.xlabel('Mean Predicted Probability (Bin)')
        plt.ylabel('Fraction of Positives (Bin)')
        plt.title(f'Calibration Curve: {label_name}\n{clf_name} (BoW Features)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(calibration_plot_path)
        plt.close()
        print(f"[INFO] Calibration plot saved to {calibration_plot_path}")
    else:
        print(f"[INFO] Skipping calibration plot for {clf_name} (no probabilities calculated or non-binary labels).")

    if vectorizer is not None:
        feature_names = vectorizer.get_feature_names_out()
        model_to_inspect = clf.estimator if isinstance(clf, CalibratedClassifierCV) else clf

        feature_importances_data = []
        if hasattr(model_to_inspect, 'coef_'):
            print(f"[INFO] Extracting coefficients for {clf_name}...")
            if model_to_inspect.coef_.ndim > 1:
                if model_to_inspect.classes_.shape[0] == 2:
                    coefs = model_to_inspect.coef_[1]
                else:
                    print(f"[WARNING] {clf_name} is a multi-class classifier with coef_. Showing absolute sum of coefficients.")
                    coefs = np.sum(np.abs(model_to_inspect.coef_), axis=0)
            else:
                coefs = model_to_inspect.coef_

            sorted_indices = np.argsort(np.abs(coefs))[::-1]
            for i in sorted_indices[:10]:
                feature_importances_data.append({
                    'feature': feature_names[i],
                    'coefficient': coefs[i],
                    'abs_coefficient': np.abs(coefs[i]),
                    'rank': len(feature_importances_data) + 1
                })
            print(f"[INFO] Top 10 coefficients for {clf_name} extracted.")

        elif hasattr(model_to_inspect, 'feature_importances_'):
            print(f"[INFO] Extracting feature importances for {clf_name}...")
            importances = model_to_inspect.feature_importances_
            sorted_indices = np.argsort(importances)[::-1]
            for i in sorted_indices[:10]:
                feature_importances_data.append({
                    'feature': feature_names[i],
                    'importance': importances[i],
                    'rank': len(feature_importances_data) + 1
                })
            print(f"[INFO] Top 10 feature importances for {clf_name} extracted.")
        else:
            print(f"[INFO] {clf_name} does not have 'coef_' or 'feature_importances_' attribute for feature importance. Skipping.")

        if feature_importances_data:
            feature_importance_df = pd.DataFrame(feature_importances_data)
            feature_importance_df.to_csv(feature_importance_path, index=False)
            print(f"[INFO] Top 10 feature importances saved to {feature_importance_path}")
        else:
            print(f"[INFO] No feature importances to save for {clf_name}.")

    metrics_data = {
        'label_name': label_name,
        'classifier_name': clf_name,
        'AUROC': auroc
    }
    if perform_hypersearch and grid_search_results:
        best_params = {f'param_{k}': v for k, v in grid_search_results.best_params_.items()}
        metrics_data.update(best_params)
        metrics_data['best_cv_score'] = grid_search_results.best_score_

    metrics_df = pd.DataFrame([metrics_data])
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"[INFO] Metrics saved to {metrics_csv_path}")
    print(f"--- Finished: {clf_name} for label '{label_name}' ---")

def aggregate_results(metrics_dir):
    INTERPRETABLE_MODELS = [
        "LinearSVC", "SGDClassifier", "RidgeClassifierCV",
        "Perceptron", "PassiveAggressiveClassifier",
        "RandomForestClassifier", "DecisionTreeClassifier"
    ]

    os.makedirs(metrics_dir, exist_ok=True)

    interpretable_file = os.path.join(metrics_dir, "best_interpretable_models.csv")

    all_metrics = []
    for fname in os.listdir(metrics_dir):
        if fname.startswith('metrics_label_') and fname.endswith(".csv"):
            try:
                df = pd.read_csv(os.path.join(metrics_dir, fname))
                all_metrics.append(df)
            except Exception as e:
                print(f"[WARNING] Could not read {fname}: {e}")

    if not all_metrics:
        print("[ERROR] No metric files found to aggregate.")
        return

    all_results = pd.concat(all_metrics, ignore_index=True)

    interpretable_results = all_results[all_results['classifier_name'].isin(INTERPRETABLE_MODELS)]
    best_interpretable = interpretable_results.loc[
        interpretable_results.groupby('label_name')['AUROC'].idxmax()
    ].sort_values('AUROC', ascending=False) if not interpretable_results.empty else pd.DataFrame()

    if not best_interpretable.empty:
        best_interpretable.to_csv(interpretable_file, index=False)
        print(f"[SUCCESS] Saved best interpretable models to {interpretable_file}")
    else:
        print("[WARNING] No interpretable model results found")

    print("\n=== Best Overall Models ===")

def main(label_index, classifier_index, mode, hypersearch, train_path, val_path):
    label_columns = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum',
        'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
        'Pneumonia', 'Pneumothorax', 'Support Devices'
    ]

    if label_index < 0 or label_index >= len(label_columns):
        print(f"[ERROR] Invalid label index: {label_index}. Must be between 0 and {len(label_columns)-1}.")
        sys.exit(1)
    label_name = label_columns[label_index]

    if classifier_index < 0 or classifier_index >= len(CLASSIFIERS):
        print(f"[ERROR] Invalid classifier index: {classifier_index}. Must be between 0 and {len(CLASSIFIERS)-1}.")
        sys.exit(1)

    output_dir = 'bow_sweep_results'
    metrics_output_dir = os.path.join(output_dir, 'metrics')

    if mode == 'aggregate':
        print(f"[INFO] Aggregating results from {metrics_output_dir}...")
        aggregate_results(metrics_output_dir)
        return

    print("[INFO] Loading datasets...")
    try:
        train_df = pd.read_csv(train_path, compression='gzip' if train_path.endswith('.gz') else None)
        val_df = pd.read_csv(val_path, compression='gzip' if val_path.endswith('.gz') else None)
    except FileNotFoundError as e:
        print(f"[ERROR] Data file not found: {e}. Please check the file paths.")
        sys.exit(1)

    print("[INFO] Preprocessing text and handling labels...")
    for df in [train_df, val_df]:
        df[label_columns] = df[label_columns].fillna(0)
        if 'text' in df.columns:
            df['processed_text'] = df['text'].apply(chunk)
        else:
            print(f"[ERROR] 'text' column not found in one of the dataframes.")
            sys.exit(1)

    print("[INFO] Creating Bag-of-Words features...")
    vectorizer = CountVectorizer(max_features=8192, max_df=0.90, min_df=5, stop_words='english', binary=False)
    if train_df['processed_text'].empty:
        print("[ERROR] Training data has no text to vectorize after preprocessing.")
        sys.exit(1)

    print("[INFO] Fitting CountVectorizer on training data...")
    train_bow = vectorizer.fit_transform(train_df['processed_text'])
    print(f"[INFO] Vocabulary size: {len(vectorizer.get_feature_names_out())}")

    print("[INFO] Transforming validation data...")
    val_bow = vectorizer.transform(val_df['processed_text'])
    print(f"[INFO] Shape of BoW matrices - Train: {train_bow.shape}, Val: {val_bow.shape}")

    train_and_evaluate(
        classifier_index=classifier_index,
        label_name=label_name,
        train_df=train_df,
        val_df=val_df,
        train_bow=train_bow,
        val_bow=val_bow,
        output_dir=output_dir,
        perform_hypersearch=hypersearch,
        vectorizer=vectorizer
    )

    print("\n[INFO] Script finished.")
    print(f"[INFO] Results saved in: {output_dir}")

    vectorizer_path = os.path.join(output_dir, 'models', 'bow_vectorizer.pkl')
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a single text classifier using Bag-of-Words features.")
    parser.add_argument('--label_index', type=int, required=True,
                        help='Index of the label column (0-13) to use for classification.')
    parser.add_argument('--classifier_index', type=int, required=True,
                        help='Index of the classifier (0-6) to run.')
    parser.add_argument('--mode', type=str, choices=['train', 'aggregate'], required=True,
                        help='Mode: "train" to run training/evaluation, "aggregate" to combine results.')
    parser.add_argument('--hypersearch', action='store_true',
                        help='Perform hyperparameter search using GridSearchCV.')
    parser.add_argument('--train_path', type=str, required=True,
                        help='Path to the training CSV file (can be gzipped).')
    parser.add_argument('--val_path', type=str, required=True,
                        help='Path to the validation CSV file (can be gzipped).')
    args = parser.parse_args()

    main(args.label_index, args.classifier_index, args.mode, args.hypersearch, 
         args.train_path, args.val_path)