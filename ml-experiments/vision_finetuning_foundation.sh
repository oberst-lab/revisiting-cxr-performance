#!/bin/bash
#SBATCH --job-name=vision_probe
#SBATCH --output=logs/probe_%A_%a.out
#SBATCH --error=logs/probe_%A_%a.err
#SBATCH --array=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=a100
#SBATCH --time=48:00:00

source ~/.bashrc
conda activate mlhc

MODELS=("densenet121" "resnet50")
INDEX=$(($SLURM_ARRAY_TASK_ID - 1))
CURRENT_MODEL=${MODELS[$INDEX]}

scontrol update JobId=$SLURM_JOB_ID JobName="vision_probe_${CURRENT_MODEL}"

# Create the checkpoints directory if it doesn't exist
mkdir -p ./checkpoints

DATA_DIR=/path/to/exp_data
IMG_PREFIX=/path/to/mimic-cxr-jpg/2.1.0/files/

# Run the training script
python vision_finetuning_foundation.py \
    --model_type ${CURRENT_MODEL} \
    --epochs 10 \
    --lr 1e-3 \
    --batch_size 256 \
    --out_dir ./checkpoints \
    --train_csv ${DATA_DIR}/train.csv.gz \
    --val_csv ${DATA_DIR}/val.csv.gz \
    --test_csv ${DATA_DIR}/test.csv.gz \
    --img_prefix ${IMG_PREFIX} \
    --num_workers 8

