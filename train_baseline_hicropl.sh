#!/bin/bash
set -e

# Experiment Name: ViTB32_BaselinePlus_HiCroPL
EXP="ViTB32_BaselinePlus_HiCroPL"
OUT="outputs/${EXP}-$(date +%m-%d-%H%M)"
mkdir -p "${OUT}"

echo "Starting Training: Baseline Configuration + Lite-HiCroPL + Consistency Loss"

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
  --gpu mps \
  --seed 42 \
  --workers 4 \
  --print-freq 10 \
  \
  --root-dir ./ \
  --train-annotation RAER/annotation/train_80.txt \
  --val-annotation RAER/annotation/val_20.txt \
  --test-annotation RAER/annotation/test.txt \
  --bounding-box-face RAER/bounding_box/face.json \
  --bounding-box-body RAER/bounding_box/body.json \
  \
  --clip-path ViT-B/32 \
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
  --batch-size 8 \
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

# Note: --use-baseline-config True in main.py will override LRs and disable complex balancing logic.
# But we still keep --use-hierarchical-prompt True to use our better prompts.
