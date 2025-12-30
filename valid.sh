#!/bin/bash

python main.py \
    --mode eval \
    --gpu mps \
    --exper-name test_eval \
    --eval-checkpoint outputs/model_best.pth \
    --root-dir . \
    --test-annotation RAER/annotation/test.txt \
    --clip-path ViT-B/16 \
    --bounding-box-face RAER/bounding_box/face.json \
    --bounding-box-body RAER/bounding_box/body.json \
    --text-type class_descriptor \
    --contexts-number 8 \
    --class-token-position end \
    --class-specific-contexts True \
    --load_and_tune_prompt_learner True \
    --temporal-layers 1 \
    --num-segments 16 \
    --duration 1 \
    --image-size 224 \
    --seed 42
