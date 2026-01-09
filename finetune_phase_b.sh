#!/bin/bash
set -e

# SCRIPT CHO PHA B: SỬ DỤNG COST-SENSITIVE LOSS
# Mục tiêu: Kế thừa các cài đặt của Pha A và thêm vào cơ chế phạt các lỗi sai
# cụ thể (Neutral -> Confusion/Distraction) để đẩy recall Neutral lên cao hơn.

# --- Cấu hình chung ---
# Checkpoint để bắt đầu finetune
# Có thể bắt đầu từ checkpoint gốc:
RESUME_CHECKPOINT="outputs/Kaggle_ViTB32_LiteHiCroPL_4Stage_SmartPush_100Epochs-Resumed-STAGE3_EXTENDED-[01-05]-[17:35]/model_best.pth"
# HOẶC từ kết quả tốt nhất của Pha A:
# RESUME_CHECKPOINT="outputs/Finetune-PhaseA-NeutralFocus/model_best.pth"

# Tên cho lần chạy mới
EXP_NAME="Finetune-PhaseB-CostSensitive"

# Số epoch để finetune
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
echo "Bắt đầu Finetune Pha B: Sử dụng Cost-Sensitive Loss"
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
  # --- THAM SỐ CỦA PHA A (kế thừa và override các giá trị gốc) ---
  --use-focal-loss False \
  --stage3_logit_adjust_tau 0.25 \
  --stage3_max_class_weight 1.8 \
  --lambda-mi 0.2 \
  --mi-warmup 0 \
  --lambda-dc 0.2 \
  --dc-warmup 0 \
  --use_amp True \
  \
  # --- ✅ THAM SỐ MỚI CHO PHA B ---
  # Kích hoạt cơ chế phạt lỗi sai và đặt mức phạt.
  # Giá trị > 1.0. Bắt đầu với 2.0 hoặc 2.5
  --cost_sensitive_beta 2.0

echo "Finetune Pha B hoàn tất."