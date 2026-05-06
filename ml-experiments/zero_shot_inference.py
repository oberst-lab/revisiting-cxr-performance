"""
zero_shot_inference.py: BiomedCLIP zero-shot multilabel classification on MIMIC-CXR test set.
                        Uses cosine similarity between image embeddings and per-label text
                        prompts as prediction scores, then evaluates with MultilabelAUROC.
"""

import torch
import os
import pandas as pd
import argparse
import torchmetrics
import tqdm
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from open_clip import create_model_from_pretrained, get_tokenizer


LABEL_COLUMNS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
    'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
    'Pneumothorax', 'Support Devices'
]

# Label phrases: use the CheXpert label name directly where natural;
# minimal rewording only for labels that are grammatically awkward as-is.
# "No Finding" follows the CheXpert definition: absence of all 13 pathology labels.
LABEL_PHRASES = {
    'Atelectasis':                'atelectasis',
    'Cardiomegaly':               'cardiomegaly',
    'Consolidation':              'consolidation',
    'Edema':                      'pulmonary edema',
    'Enlarged Cardiomediastinum': 'enlarged cardiomediastinum',
    'Fracture':                   'fracture',
    'Lung Lesion':                'lung lesion',
    'Lung Opacity':               'lung opacity',
    'No Finding':                 'no acute cardiopulmonary findings',
    'Pleural Effusion':           'pleural effusion',
    'Pleural Other':              'pleural abnormality',
    'Pneumonia':                  'pneumonia',
    'Pneumothorax':               'pneumothorax',
    'Support Devices':            'support devices',
}

# Multiple surface-form templates — embeddings are averaged per label (prompt ensemble).
# Improves robustness against prompt wording sensitivity.
TEMPLATES = [
    'This chest radiograph demonstrates {}.',
    'Chest X-ray findings consistent with {}.',
    'This CXR shows evidence of {}.',
    'Radiographic appearance of {} on chest X-ray.',
]


class CXRDataset(Dataset):
    def __init__(self, df, preprocess):
        self.paths = df['images'].tolist()
        self.labels = df[LABEL_COLUMNS].values.astype(float)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        img = self.preprocess(img)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_csv',   type=str, required=True)
    parser.add_argument('--img_prefix', type=str, default='')
    parser.add_argument('--out_dir',    type=str, default='./checkpoints/biomedclip_zeroshot')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers',type=int, default=8)
    parser.add_argument('--model_name', type=str,
                        default='hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading BiomedCLIP from {args.model_name} ...")
    model, preprocess = create_model_from_pretrained(args.model_name)
    tokenizer = get_tokenizer(args.model_name)
    model = model.to(device).eval()

    # Encode text prompts once — ensemble over templates, average per label
    print("Encoding text prompts ...")
    with torch.inference_mode():
        all_template_features = []
        for template in TEMPLATES:
            prompts = [template.format(LABEL_PHRASES[lbl]) for lbl in LABEL_COLUMNS]
            tokens = tokenizer(prompts).to(device)
            feats = model.encode_text(tokens)                          # (14, D)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_template_features.append(feats)
        # Average across templates then re-normalise
        text_features = torch.stack(all_template_features).mean(dim=0)  # (14, D)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Load test set
    print("Loading test CSV ...")
    test_df = pd.read_csv(args.test_csv, compression='gzip')
    test_df['images'] = (
        args.img_prefix
        + 'p' + test_df['subject_id'].astype(str).str[:2]
        + '/p' + test_df['subject_id'].astype(str)
        + '/s' + test_df['study_id'].astype(str)
        + '/' + test_df['dicom_id'].astype(str) + '.jpg'
    )
    test_df[LABEL_COLUMNS] = test_df[LABEL_COLUMNS].fillna(0)

    dataset = CXRDataset(test_df, preprocess)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    auroc_macro    = torchmetrics.classification.MultilabelAUROC(
                        num_labels=len(LABEL_COLUMNS), average='macro').to(device)
    auroc_perclass = torchmetrics.classification.MultilabelAUROC(
                        num_labels=len(LABEL_COLUMNS), average='none').to(device)

    print("Running zero-shot inference ...")
    with torch.inference_mode():
        for images, labels in tqdm.tqdm(loader, desc="Test"):
            images = images.to(device)
            labels = labels.to(device).long()
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            # cosine similarity → use as logit/score (no sigmoid needed for AUROC)
            scores = (image_features @ text_features.T).float()  # (B, 14)
            auroc_macro.update(scores, labels)
            auroc_perclass.update(scores, labels)

    final_macro      = auroc_macro.compute().item()
    final_per_class  = auroc_perclass.compute().cpu().numpy()

    print(f"\nBiomedCLIP Zero-Shot Test Results:")
    print(f"  Macro AUROC: {final_macro:.4f}")
    results = []
    for idx, label in enumerate(LABEL_COLUMNS):
        score = final_per_class[idx]
        print(f"  {label}: {score:.4f}")
        results.append({'Label': label, 'Test_AUROC': score})
    results.append({'Label': 'MACRO_AVERAGE', 'Test_AUROC': final_macro})

    out_path = os.path.join(args.out_dir, 'test_metrics.csv')
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nMetrics saved to {out_path}")


if __name__ == '__main__':
    main()
