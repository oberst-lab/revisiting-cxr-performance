import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron, RidgeClassifierCV, PassiveAggressiveClassifier, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
import torch
import sklearn
from sklearn.utils.class_weight import compute_sample_weight
from inspect import signature
sklearn.set_config(enable_metadata_routing=True) # this is for routing enabling for cross fold validation grouping

"""
Output folders:
    /text_classifiers_metrics_train_val_split - save evaluation metrics for each classifier
    /text_models - save trained models
"""

CLASSIFIERS = [
    ("Perceptron", Perceptron(penalty='l2', random_state=42)),
    ("RidgeClassifierCV", RidgeClassifierCV(alphas=np.logspace(-3, 3, 7))),
    ("PassiveAggressiveClassifier", PassiveAggressiveClassifier(random_state=42)),
    ("GaussianNB", GaussianNB()),
    ("LinearDiscriminantAnalysis", LinearDiscriminantAnalysis(shrinkage='auto', solver='lsqr')),
    ("RandomForestClassifier", RandomForestClassifier(random_state=42)),
    ("KNeighborsClassifier", KNeighborsClassifier()),
    ("AdaBoostClassifier", AdaBoostClassifier(random_state=42)),
    ("MLPClassifier", MLPClassifier(early_stopping=True, validation_fraction=0.1, max_iter=300, random_state=42)),
    ("LinearSVC", LinearSVC(max_iter=1000, random_state=42)),
    ("SGDClassifier", SGDClassifier(loss='log_loss', penalty='l2', random_state=42))
]

PARAM_GRIDS = {
    "Perceptron": {'alpha': [1e-5, 1e-4, 1e-3]},
    "RidgeClassifierCV": {'alphas': [np.logspace(-4, 4, 9)]},
    "PassiveAggressiveClassifier": {'C': [0.01, 0.1, 0.5, 1.0]},
    "GaussianNB": {},
    "LinearDiscriminantAnalysis": {'solver': ['lsqr', 'eigen'], 'shrinkage': [None, 'auto']},
    "RandomForestClassifier": {'n_estimators': [50, 100, 150], 'max_depth': [None, 5, 10], 'min_samples_leaf': [1, 5, 10]},
    "KNeighborsClassifier": {'n_neighbors': [3, 5, 7, 9]},
    "AdaBoostClassifier": {'n_estimators': [50, 100], 'learning_rate': [0.1, 0.5, 1.0]},
    "MLPClassifier": {'alpha': [1e-5, 1e-4, 1e-3], 'hidden_layer_sizes': [(64,), (128,), (64, 32)]},
    "LinearSVC": {'C': [0.1, 1.0, 10.0]},
    "SGDClassifier": {'alpha': [1e-5, 1e-4, 1e-3], 'penalty': ['l1', 'l2']}
}


def train_and_evaluate(classifier_index, label_name, train_df, val_df,
                       train_embeddings, val_embeddings, output_dir, perform_hypersearch=False):
    clf_name, base_clf = CLASSIFIERS[classifier_index]
    print(f"[INFO] Training and calibrating {clf_name} on train set for '{label_name}'...")

    model_path = f'text_models/{label_name}_{clf_name}_train_val_split.pkl'
    calibration_plot_path = os.path.join(output_dir, f'calibration_plot_{label_name}_{clf_name}_train_val_split.pdf')
    metrics_csv_path = os.path.join(output_dir, f'metrics_label_{label_name}_{clf_name}_train_val_split.csv')

    train_labels = train_df[label_name].values
    train_groups = train_df["subject_id"].values

    # Ensure dense arrays if needed
    if isinstance(base_clf, (GaussianNB, LinearDiscriminantAnalysis)):
        train_embeddings = train_embeddings.toarray() if hasattr(train_embeddings, "toarray") else train_embeddings
        val_embeddings = val_embeddings.toarray() if hasattr(val_embeddings, "toarray") else val_embeddings

    # Check if the base classifier accepts sample_weight
    if hasattr(base_clf, 'fit') and 'sample_weight' in signature(base_clf.fit).parameters:
        best_clf = base_clf.set_fit_request(sample_weight=True)
    else:
        best_clf = base_clf
    
    sample_weight = compute_sample_weight('balanced', y=train_labels)
    
    if perform_hypersearch and clf_name in PARAM_GRIDS:
        print(f"[INFO] Performing hyperparameter search (StratifiedGroupKFold) for {clf_name}...")
        param_grid = PARAM_GRIDS[clf_name]
        group_kfold = StratifiedGroupKFold(n_splits=5)
        grid_search = GridSearchCV(base_clf, param_grid, cv=group_kfold, scoring='roc_auc', n_jobs=-1, verbose=1)
        grid_search.fit(train_embeddings, train_labels, groups=train_groups)
        best_clf = grid_search.best_estimator_
        print(f"[INFO] Best hyperparameters for {clf_name}: {grid_search.best_params_}")

    # Calibrate using StratifiedGroupKFold on training set only
    print(f"[INFO] Calibrating the model for {clf_name} on train set...")
    group_kfold = StratifiedGroupKFold(n_splits=5)
    clf = CalibratedClassifierCV(estimator=best_clf, cv=group_kfold, method='sigmoid')
    clf.fit(train_embeddings, train_labels, groups=train_groups)

    # Save the calibrated model
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)

    # Predict on validation set (for model selection)
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(val_embeddings)[:, 1]
    else:
        y_prob = clf.decision_function(val_embeddings)
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())  # normalize

    val_labels = val_df[label_name].values
    auroc = metrics.roc_auc_score(val_labels, y_prob)
    prob_true, prob_pred = calibration_curve(val_labels, y_prob, n_bins=10)

    # Plot calibration curve
    plt.figure(figsize=(10, 6))
    plt.plot(prob_pred, prob_true, marker='o', label=f'{clf_name}', color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title(f'Calibration Curve: {label_name} ({clf_name}) - Train/Val Split')
    plt.legend()
    plt.grid()
    plt.savefig(calibration_plot_path)
    plt.close()

    # Save AUROC and potentially best hyperparameters
    metrics_data = {
        'label_name': label_name,
        'classifier_name': clf_name,
        'AUROC': auroc
    }
    if perform_hypersearch and clf_name in PARAM_GRIDS and hasattr(grid_search, 'best_params_'):
        metrics_data.update(grid_search.best_params_)

    metrics_df = pd.DataFrame([metrics_data])
    metrics_df.to_csv(metrics_csv_path, index=False)


def aggregate_results(metrics_dir='text_classifiers_metrics_train_val_split/'):
    all_metrics = []
    for fname in os.listdir(metrics_dir):
        if fname.endswith(".csv"):
            df = pd.read_csv(os.path.join(metrics_dir, fname))
            all_metrics.append(df)
    result = pd.concat(all_metrics)
    result.sort_values(by=['classifier_name'], inplace=True)
    result.to_csv("text_classifiers_metrics_train_val_split/all_results.csv", index=False)
    print(f"[INFO] Aggregated {len(all_metrics)} results into all_results.csv")


def main(label_index, classifier_index, mode, hypersearch, train_csv_path, val_csv_path, 
         train_embeddings_path, val_embeddings_path):
    label_columns = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
                     'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
                     'Pneumonia', 'Pneumothorax', 'Support Devices']

    label_name = label_columns[label_index]

    if mode == 'aggregate':
        aggregate_results('text_classifiers_metrics_train_val_split/')
        return

    print("[INFO] Loading datasets...")
    train_df = pd.read_csv(train_csv_path, compression='gzip' if train_csv_path.endswith('.gz') else None)
    val_df = pd.read_csv(val_csv_path, compression='gzip' if val_csv_path.endswith('.gz') else None)

    for df in [train_df, val_df]:
        df[label_columns] = df[label_columns].fillna(0)

    # Using proper train/val split for model selection
    print("[INFO] Using proper train/val split for model selection...")

    print("[INFO] Loading embeddings...")
    train_embeddings = torch.load(train_embeddings_path)
    val_embeddings = torch.load(val_embeddings_path)

    # Convert to numpy if they are tensors (for sklearn compatibility)
    if isinstance(train_embeddings, torch.Tensor):
        train_embeddings = train_embeddings.numpy()
    if isinstance(val_embeddings, torch.Tensor):
        val_embeddings = val_embeddings.numpy()

    os.makedirs('text_classifiers_metrics_train_val_split/', exist_ok=True)
    os.makedirs('text_models/', exist_ok=True)

    # Now using train for training, val for model selection (compliant approach)
    train_and_evaluate(
        classifier_index=classifier_index,
        label_name=label_name,
        train_df=train_df,           # Only train data for training
        val_df=val_df,               # Val data for evaluation/model selection
        train_embeddings=train_embeddings,     # Only train embeddings
        val_embeddings=val_embeddings,         # Val embeddings for evaluation
        output_dir='text_classifiers_metrics_train_val_split/',
        perform_hypersearch=hypersearch
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train text classifiers on MIMIC-CXR embeddings with compliant train/val split.")
    parser.add_argument('--label_index', type=int, required=True, help='Index of the label to evaluate')
    parser.add_argument('--classifier_index', type=int, required=False, default=8, help='Index into the list of classifiers')
    parser.add_argument('--mode', type=str, choices=['train', 'aggregate'], required=True, help='Mode: train or aggregate')
    parser.add_argument('--hypersearch', action='store_true', help='Perform hyperparameter search using GridSearchCV on the train set with StratifiedGroupKFold.')
    
    # Add path arguments
    parser.add_argument('--train_csv', type=str, required=True, help='Path to training CSV file')
    parser.add_argument('--val_csv', type=str, required=True, help='Path to validation CSV file')
    parser.add_argument('--train_embeddings', type=str, required=True, help='Path to training embeddings file')
    parser.add_argument('--val_embeddings', type=str, required=True, help='Path to validation embeddings file')
    
    args = parser.parse_args()
    main(args.label_index, args.classifier_index, args.mode, args.hypersearch,
         args.train_csv, args.val_csv, args.train_embeddings, args.val_embeddings)