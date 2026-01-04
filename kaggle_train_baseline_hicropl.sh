#!/bin/bash
set -e

# --- SETUP ENVIRONMENT (Kaggle/Colab) ---
echo "=> Installing dependencies..."
pip install git+https://github.com/openai/CLIP.git
pip install imbalanced-learn

# Experiment Name
EXP="Kaggle_ViTB32_BaselinePlus_HiCroPL"
OUT="outputs/${EXP}-$(date +%m-%d-%H%M)"
mkdir -p "${OUT}"

echo "Starting Training on Kaggle: Baseline Configuration + Lite-HiCroPL + Consistency Loss"

# --- PATH CONFIGURATION ---
# Adjust these paths if your Kaggle dataset structure is different
# Root dir containing the 'RAER' folder (or where video paths start)
ROOT_DIR="/kaggle/input/raer-video-emotion-dataset"

# Annotation paths
ANNOT_DIR="/kaggle/input/raer-annot/annotation"
TRAIN_TXT="${ANNOT_DIR}/train_80.txt"
VAL_TXT="${ANNOT_DIR}/val_20.txt"
TEST_TXT="${ANNOT_DIR}/test.txt"

# Bounding Box paths (Corrected based on user input for main dataset)
BOX_DIR="${ROOT_DIR}/RAER/bounding_box"
FACE_BOX="${BOX_DIR}/face.json"
BODY_BOX="${BOX_DIR}/body.json"

# CLIP Model Path (Kaggle usually has internet, so ViT-B/32 works. 
# If offline, upload the .pt file and point to it)
CLIP_PATH="ViT-B/32"

# Baseline Config:
# - LR: 0.01 (High)
# - Image Encoder LR: 1e-5 (High)
# - Contexts: 8 (Low complexity)
# - No Weighted Sampler
# - No Logit Adjustment
# - No Focal Loss (Standard CE)

python main.py \
  --mode train \
  --exper-name "${EXP}" \
  --gpu 0 \
  --seed 42 \
  --workers 4 \
  --print-freq 50 \
  \
  --root-dir "${ROOT_DIR}" \
  --train-annotation "${TRAIN_TXT}" \
  --val-annotation "${VAL_TXT}" \
  --test-annotation "${TEST_TXT}" \
  --bounding-box-face "${FACE_BOX}" \
  --bounding-box-body "${BODY_BOX}" \
  \
  --clip-path "${CLIP_PATH}" \
  --text-type class_descriptor \
  --image-size 224 \
  --num-segments 16 \
  --duration 1 \
  --temporal-layers 1 \
  --contexts-number 8 \
  \
  --use-hierarchical-prompt True \
  \
  --epochs 60 \
  --batch-size 16 \
  \
  --lr 0.01 \
  --lr-image-encoder 1e-5 \
  --lr-prompt-learner 0.001 \
  --milestones 30 50 \
  --gamma 0.1 \
  \
  --lambda-mi 0.0 \
  --lambda-dc 0.0 \
  --lambda-cons 0.1 \
  \
  --semantic-smoothing True \
  --use-focal-loss False \
  --unfreeze-visual-last-layer False \
  \
  --use-baseline-config True

# Note: Batch size increased to 16 for Kaggle GPUs.
