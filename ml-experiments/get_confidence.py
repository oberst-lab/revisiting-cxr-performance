import os
import pickle
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import encoding_utils
import torch

def remove_non_best_models(metrics_df):
    """Removes .pkl files from 'text_models' that are not the best performing for each label."""
    best_models = metrics_df.sort_values(by='AUROC', ascending=False).drop_duplicates('label_name')
    best_model_files = set()
    for index, row in best_models.iterrows():
        label_name = row['label_name']
        clf_name = row['classifier_name']
        best_model_files.add(f"{label_name}_{clf_name}_full_train.pkl")

    model_dir = 'text_models'
    if os.path.exists(model_dir):
        for filename in os.listdir(model_dir):
            if filename.endswith(".pkl") and filename not in best_model_files:
                file_path = os.path.join(model_dir, filename)
                try:
                    os.remove(file_path)
                    print(f"[INFO] Removed non-best model: {filename}")
                except Exception as e:
                    print(f"[ERROR] Could not remove {filename}: {e}")
    else:
        print(f"[WARNING] Directory '{model_dir}' not found.")


def main():
    # Load the metrics file
    metrics_df = pd.read_csv('text_classifiers_metrics_full_train/all_results.csv')
    test_embeddings = torch.load('experiments/test.pt')
    test_df = pd.read_csv('experiments/experiments.test.csv.gz', compression='gzip')

    label_columns = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
                         'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
                         'Pneumonia', 'Pneumothorax', 'Support Devices']
    for label_index in range(len(label_columns)):
        label_name = label_columns[label_index]
        test_df[label_columns] = test_df[label_columns].fillna(0)
        test_labels = test_df[label_name].values
        os.makedirs('confident_predictions_test', exist_ok=True)
        # Get the best classifier per label
        best_df = metrics_df.sort_values(by='AUROC', ascending=False).drop_duplicates('label_name').copy()
        best_clf_name = str(best_df[best_df['label_name'] == label_name]['classifier_name'].values[0])
        model_path = f"text_models/{label_name}_{best_clf_name}_full_train.pkl"
        encoding_utils.evaluate_classifier(model_path=model_path,
                                            label_name=label_name,
                                            labels=test_labels,
                                            embeddings=test_embeddings,
                                            output_dir='confident_predictions_test',
                                            mode='test')

    # Remove non-best performing models
    remove_non_best_models(metrics_df)
    best_df = best_df.iloc[:,:3].rename(columns={'label_name':'Label', 'classifier_name':'Classifer'})
    best_df['AUROC'] = best_df['AUROC'].apply(lambda x: float(f"{x:.3g}"))
    best_df.to_csv('plots/best_text_classifiers.csv', index=False)

if __name__ == '__main__':
    main()
