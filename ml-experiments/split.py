"""
split.py: Creates the train/val/test split on the processed data to be used for
          all downstream tasks. Uses a randomized split based on the unique subject
          ids instead of the canonical/provided split from MIMIC documentation, because
          of the potential issues involved in splitting on subjects with multiple
          images/notes in our analyses. Additionally, groups all notes associated with
          a given image (dicom_id) in this process.

Usage:
python split.py - writes splits to be used downstream to data/experiments/*.csv.gz
"""
import numpy as np  
import pandas as pd  
import encoding_utils

def main():
    merged_df = pd.read_csv('experiments/xray_all_labels.csv.gz', compression='gzip')
    df_grouped = merged_df.groupby('dicom_id').agg(lambda x: 'eotextdelimiter'.join(x.unique()) if x.name == 'text' else x.iloc[0]).reset_index()
    df_grouped['text'] = encoding_utils.text_preprocessing(df_grouped['text'])

    print("Saving to IMAGES_DOWN...\n")
    # Save the unique values to a file called IMAGES_DOWN
    with open('IMAGES_DOWN', 'w') as f:
        for dicom_id in df_grouped['images'].values:
            f.write(f"{dicom_id}\n")


    # Now, to create the train/val/test split, based on the unique subjects
    unique_patients = df_grouped['subject_id'].unique()
    # Split patients
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(unique_patients)
    train_split = int(0.8 * len(unique_patients))
    val_split = int(0.9 * len(unique_patients))
    test_split = len(unique_patients) - val_split - train_split
    train_patients = unique_patients[:train_split]
    val_patients = unique_patients[train_split:val_split]
    test_patients = unique_patients[val_split:]
    train_export = df_grouped[df_grouped['subject_id'].isin(train_patients)]
    val_export = df_grouped[df_grouped['subject_id'].isin(val_patients)]
    test_export = df_grouped[df_grouped['subject_id'].isin(test_patients)]
    train_export.to_csv('experiments/train.csv.gz', index=False, compression='gzip')
    val_export.to_csv('experiments/val.csv.gz', index=False, compression='gzip')
    test_export.to_csv('experiments.test.csv.gz', index=False, compression='gzip')

if __name__ == "__main__":
    main()
