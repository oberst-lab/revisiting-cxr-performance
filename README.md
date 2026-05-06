# Revisiting Performance Claims for Chest X-Ray Models Using Clinical Context

## Data

Requires credentialed access to [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/) and [MIMIC-IV](https://physionet.org/content/mimiciv/) via PhysioNet.

## Setup

```bash
conda env create -f mlhc.yml
conda activate mlhc
```

## Pipeline

Scripts are in `ml-experiments/` and follow this order:

1. **Data preparation**: `create_experiment_data.py` → `split.py`
2. **Vision model**: `imagenet_distributed.py` → `vision_evaluation.py` → `vision_calibration.py`
3. **Text classifiers** (pre-test probability): `generate_embeddings.py` + `sweep_text.py` for LM-based; `bag_of_words.py` for BoW-based
4. **Stratified analysis**: `get_quantiles.py` → `quantile_analysis.py`; `prior_mention_list_eval.py`
5. **Matched / reweighted analysis**: `matching_neighbors.py` → `neighbor_diffs.py`

Shared utilities: `encoding_utils.py`. BoW variants of scripts are suffixed `_bow`.

