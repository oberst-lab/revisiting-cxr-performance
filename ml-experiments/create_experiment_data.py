"""
create_experiment_data.py: Joins together admissions, chexpert labels, x-ray metadata, and medical notes
                           into one table, then filters for x-rays that only occur after an admission
                           discharge and associated medical note(s).

Usage:
python create_experiment_data.py - writes table to be used downstream to data/experiments/xray_all_labels.csv.gz
"""

import numpy as np  
import pandas as pd  
import encoding_utils

def main():
    
    # Read in the relevant data tables
    admissions = pd.read_csv('hosp/admissions.csv.gz', compression='gzip')
    cxr_chexpert_labels = pd.read_csv('mimic-cxr-jpg/metadata/mimic-cxr-2.0.0-chexpert.csv.gz', compression='gzip')
    cxr_meta = pd.read_csv('mimic-cxr-jpg/metadata/mimic-cxr-2.0.0-metadata.csv.gz', compression='gzip')

    # Examine number of unique subjects
    print(f"Number of unique subjects in admissions : {admissions['subject_id'].nunique()}")                                        #223,452
    print(f"Number of unique subjects in cxr_chexpert_labels : {cxr_chexpert_labels['subject_id'].nunique()}")                      #65,379
    print(f"Number of unique subjects in cxr_meta : {cxr_meta['subject_id'].nunique()}")                                            #65,379

    # Examine number of unique studies
    print(f"Number of unique studies in cxr_chexpert_labels : {cxr_chexpert_labels['study_id'].nunique()}")                         #227,827
    print(f"Number of unique studies in cxr_meta : {cxr_meta['study_id'].nunique()}")                                               #227,835

    # Merge X-Ray labels with admissions. We would expect the same number of unique subjects as in the labels (65,379)
    admission_info = admissions[['subject_id', 'hadm_id', 'admittime', 'dischtime']]
    admission_labels = pd.merge(admission_info, cxr_chexpert_labels, on='subject_id', how='inner')
    print(f"Number of unique subjects in labels/admissions : {admission_labels['subject_id'].nunique()}")                           #51,851
    # We observe that we actually have (65379 - 51851 = 13528) fewer unique subjects than we expected
    # Could these simply be subjects in the labels table that aren't in the admissions table?
    missing_subject_ids = cxr_chexpert_labels[~cxr_chexpert_labels['subject_id'].isin(admission_info['subject_id'])]
    print(f"Number of unique subjects in labels, but not in admission: {missing_subject_ids['subject_id'].nunique()}")              #13,528
    # It looks like we were right about the disparity

    # Add the metadata to this table. We would expect the same number of unique subjects (51,851) as in
    # the joined admissions and labels table, and the same number of studies as in the metadata (227,835)
    cxr_meta = cxr_meta[['subject_id', 'study_id', 'dicom_id', 'StudyTime', 'StudyDate']]
    final_joined = pd.merge(cxr_meta, admission_labels, on=['subject_id', 'study_id'], how='inner')
    print(f"Number of unique subjects in final table : {final_joined['subject_id'].nunique()}")                                     #51,851
    print(f"Number of unique studies in final table : {final_joined['study_id'].nunique()}")                                        #204,036
    # We observe that we have the same number of unique subjects, but we dropped (227835 - 204036 = 23799)
    # studies from the metadata. Could this simply be studies in the metadata that aren't in the
    # joined admission and labels table?
    missing_study_ids = cxr_meta[~cxr_meta['study_id'].isin(admission_labels['study_id'])]
    print(f"Number of unique studies in metadata, but not the labels/admission table : {missing_study_ids['study_id'].nunique()}")  #23,799
    # It looks like we were right again about the disparity

    # Read in the associated notes
    notes = pd.read_csv('mimicnote/discharge.csv.gz', compression='gzip')
    notes = notes[['subject_id', 'hadm_id', 'charttime', 'text']]

    # Examine the number of unique subjects
    print(f"Number of unique subjects in notes : {notes['subject_id'].nunique()}")                                                  #145,914
    # Examine the number of unique admissions
    print(f"Number of unique admissions in notes : {notes['hadm_id'].nunique()}")                                                   #331,793


    # Now since we had 51,851 unique subjects in our earlier table, we should hope to retain that number after we join
    # the notes.
    # How many unique admissions did we have in our earlier table?
    print(f"Number of unique admissions in table : {final_joined['hadm_id'].nunique()}")                                            #219,727
    # We would expect to retain this number as well

    # Join the notes with the table from before, matching on both subject and hadm ids:
    merged_df = pd.merge(final_joined, notes, on=['subject_id', 'hadm_id'], how='inner')
    print(f"Number of unique subjects in final table : {merged_df['subject_id'].nunique()}")                                        #45,921
    print(f"Number of unique admissions in final table : {merged_df['hadm_id'].nunique()}")                                         #162,996

    # We look to have dropped (219727 - 162996 = 56,731) unique admissions. Where did they go?
    # Maybe we had admissions in our original table that we simply don't have notes for.
    # Let's check this:
    missing_hadm_ids = final_joined[~final_joined['hadm_id'].isin(notes['hadm_id'])]
    print(f"Number of unique admissions in original table without notes: {missing_hadm_ids['hadm_id'].nunique()}")                  #56,731
    # Looks good to me

    # Hm, looks like we also dropped (51851 - 45921 = 5930) unique subjects. How did this happen?
    # Maybe we had subjects in our original table that we simply don't have notes for.
    # Let's check this:
    missing_subject_ids = final_joined[~final_joined['subject_id'].isin(notes['subject_id'])]
    print(f"Number of unique subjects in original table without notes: {missing_subject_ids['subject_id'].nunique()}")              #5,921
    # Ok, it looks like we still have 9 subjects unaccounted for. Let's track them down
    # We know that of the 5930 subjects we dropped when we added the notes, 5921 of them were dropped because they
    # simply didn't have an associated note. However, we also know that every single admission per subject that
    # didn't have an associated note was dropped as well. This implies that we have subjects that DID have a note,
    # but were dropped for some other reason

    og_dropped = final_joined[~final_joined['subject_id'].isin(merged_df['subject_id'])]
    extra_nine = og_dropped[~og_dropped['subject_id'].isin(missing_subject_ids['subject_id'])]
    print(f"Number of unique subjects not accounted for: {extra_nine['subject_id'].nunique()}")                                     #9
    # What is about these subjects that prevented them from making it to the final table? Well, let's first
    # confirm that they actually did have a note.
    confirm_note = extra_nine[~extra_nine['subject_id'].isin(notes['subject_id'])]
    assert(len(confirm_note) == 0)
    # Ok, now let's confirm that they did have an associated hadm_id in the notes table
    confirm_hadm = extra_nine[~extra_nine['hadm_id'].isin(notes['hadm_id'])]
    print(f"Number of unique admissions not accounted for: {confirm_hadm['hadm_id'].nunique()}")                                     #9
    # Aha, this is a new piece of information. We have subjects who have admissions that we simply dont have notes
    # for, which is slightly different than what we originally thought.

    # To summarize, we dropped 5930 unique subjects, where 5921 of them simply didn't exist in our notes table.
    # Then, we observe that we also dropped 56731 admissions that we also didn't have notes for. Within these,
    # we had 9 subjects with 9 admissions such that we actually have their associated subjects in the notes table,
    # just not the notes for those particular admissions.

    # Now, let's do some preprocessing/filtering. For our purposes, we are only really interested
    # in the notes with a chart time that occurs before the associated study/X-Ray, so we'll
    # want to filter to these cases. Additionally, we'll need to do some label processing (details
    # in the encoding_utils.py file

    print("Filtering/Processing the raw data...\n")

    df_processed = encoding_utils.process_data(merged_df, labels=['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
           'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
           'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
           'Pneumothorax', 'Support Devices'])


    df_processed['images'] = 'files/p' + (df_processed['subject_id'].astype(str)).str[:2] + '/p' + df_processed['subject_id'].astype(str) + '/s' + df_processed['study_id'].astype(str) + '/' + df_processed['dicom_id'].astype(str) + '.jpg'
 

    print('Saving to experiments/xray_all_labels.csv.gz...\n')

    df_processed.to_csv('experiments/xray_all_labels.csv.gz', index=False, compression='gzip')

    print('Done!')


if __name__ == "__main__":
    main()
