#!/bin/bash
set -e

# SCRIPT CHO PHA A: TẬP TRUNG KÉO RECALL LỚP NEUTRAL
# Mục tiêu: Tắt Focal Loss, giảm Logit Adjustment và Class Weighting.

# --- Cấu hình chung ---
# Checkpoint để bắt đầu finetune (sử dụng model tốt nhất từ lần chạy trước)
RESUME_CHECKPOINT="/kaggle/input/model-4stage/Kaggle_ViTB32_LiteHiCroPL_4Stage_SmartPush_100Epochs-V1/model_best.pth"

# Tên cho lần chạy mới để lưu log và model
EXP_NAME="Finetune-PhaseA-NeutralFocus"

# Số epoch để finetune (bắt đầu với 15-20 epochs)
NUM_EPOCHS=20

# --- Cấu hình PATH (CẦN CẬP NHẬT CHO MÔI TRƯỜNG CỦA BẠN) ---
# Đây là các đường dẫn ví dụ từ cấu hình Kaggle. VUI LÒNG CẬP NHẬT chúng.
ROOT_DIR="/kaggle/input/raer-video-emotion-dataset"
ANNOT_DIR="/kaggle/input/raer-annot/annotation"
TRAIN_TXT="${ANNOT_DIR}/train_80.txt"
VAL_TXT="${ANNOT_DIR}/val_20.txt"
TEST_TXT="${ANNOT_DIR}/test.txt"
BOX_DIR="${ROOT_DIR}/RAER/bounding_box"
FACE_BOX="${BOX_DIR}/face.json"
BODY_BOX="${BOX_DIR}/body.json"

# --- Lệnh thực thi ---
echo "Bắt đầu Finetune Pha A: Tập trung vào Neutral Recall"
echo "Checkpoint: $RESUME_CHECKPOINT"
echo "Lưu vào thư mục: outputs/$EXP_NAME"

python main.py \
  --mode train \
  --exper-name "${EXP_NAME}" \
  --gpu 0 \
  --seed 42 \
  --workers 4 \
  --print-freq 50 \
  --resume "${RESUME_CHECKPOINT}" \
  \
  --root-dir "${ROOT_DIR}" \
  --train-annotation "${TRAIN_TXT}" \
  --val-annotation "${VAL_TXT}" \
  --test-annotation "${TEST_TXT}" \
  --bounding-box-face "${FACE_BOX}" \
  --bounding-box-body "${BODY_BOX}" \
  \
  --clip-path "ViT-B/32" \
  --text-type class_descriptor \
  --image-size 224 \
  --num-segments 16 \
  --duration 1 \
  --temporal-layers 2 \
  --contexts-number 16 \
  --use-hierarchical-prompt True \
  \
  --epochs $NUM_EPOCHS \
  --batch-size 16 \
  \
  --lr 1e-3 \
  --lr-image-encoder 1e-6 \
  --lr-prompt-learner 5e-4 \
  --milestones 70 90 \
  --gamma 0.1 \
  \
  --lambda-cons 0.1 \
  \
  --use-lsr2-loss True \
  \
  --semantic-smoothing True \
  --focal-gamma 2.0 \
  --unfreeze-visual-last-layer False \
  \
  --stage1-epochs 3 \
  --stage1-label-smoothing 0.05 \
  --stage1-smoothing-temp 0.15 \
  \
  --stage2-epochs 30 \
  --stage2-logit-adjust-tau 0.5 \
  --stage2-max-class-weight 2.0 \
  --stage2-smoothing-temp 0.15 \
  --stage2-label-smoothing 0.1 \
  \
  --stage3-epochs 70 \
  --stage3-smoothing-temp 0.18 \
  \
  --stage4-logit-adjust-tau 0.8 \
  --stage4-max-class-weight 5.0 \
  \
  # --- THAM SỐ CỐT LÕI CHO PHA A (override các giá trị gốc) ---
  \
  # 1. Tắt Focal Loss để model không "bỏ qua" lớp Neutral
  --use-focal-loss False \
  \
  # 2. Giảm Logit Adjustment để bớt "ép" model khỏi lớp đa số
  --stage3_logit_adjust_tau 0.25 \
  \
  # 3. Giảm Class Weighting/Sampler Weight để cân bằng lại việc lấy mẫu
  --stage3_max_class_weight 1.8 \
  \
  # 4. Giảm nhẹ loss phụ để tập trung vào loss chính
  --lambda-mi 0.2 \
  --mi-warmup 0 \
  --lambda-dc 0.2 \
  --dc-warmup 0 \
  \
  --use_amp True # Đảm bảo AMP được bật (nếu dùng)

echo "Finetune Pha A hoàn tất."