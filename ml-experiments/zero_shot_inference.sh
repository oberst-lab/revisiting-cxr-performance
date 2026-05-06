#!/bin/bash
#SBATCH --job-name=biomedclip_zeroshot
#SBATCH --output=logs/zeroshot_%j.out
#SBATCH --error=logs/zeroshot_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=a100
#SBATCH --time=24:00:00

source ~/.bashrc
conda activate mlhc

mkdir -p ./checkpoints/biomedclip_zeroshot

DATA_DIR=/path/to/exp_data
IMG_PREFIX=/path/to/mimic-cxr-jpg/2.1.0/files/

python zero_shot_inference.py \
    --test_csv   ${DATA_DIR}/test.csv.gz \
    --img_prefix ${IMG_PREFIX} \
    --out_dir    ./checkpoints/biomedclip_zeroshot \
    --batch_size 256 \
    --num_workers 8
