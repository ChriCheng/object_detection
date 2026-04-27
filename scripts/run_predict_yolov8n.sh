#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ "${DEVICE_TARGET:-CPU}" = "GPU" ] && [ -f "$PROJECT_ROOT/scripts/env_mindspore_gpu.sh" ]; then
  source "$PROJECT_ROOT/scripts/env_mindspore_gpu.sh"
fi

MINDYOLO_DIR="$PROJECT_ROOT/third_party/mindyolo"
DEVICE_TARGET="${DEVICE_TARGET:-CPU}"
WEIGHT="$PROJECT_ROOT/weights/yolov8n.ckpt"
DEFAULT_IMAGE_DIR="$PROJECT_ROOT/data/coco_mindyolo/coco/images/val2017"
LOG_FILE="$PROJECT_ROOT/logs/predict_yolov8n.log"

mkdir -p "$PROJECT_ROOT/logs" "$PROJECT_ROOT/results/detect_results"

if [ ! -d "$MINDYOLO_DIR" ]; then
  echo "[ERROR] MindYOLO not found. Run: bash scripts/setup_mindyolo.sh" >&2
  exit 1
fi

if [ ! -f "$WEIGHT" ]; then
  echo "[ERROR] Weight not found: $WEIGHT. Run: bash scripts/download_yolov8n_weight.sh" >&2
  exit 1
fi

# MindYOLO demo/predict.py expects a single image file, not an image directory.
if [ -n "${IMAGE_PATH:-}" ]; then
  INPUT_IMAGE="$IMAGE_PATH"
else
  INPUT_IMAGE="$(find "$DEFAULT_IMAGE_DIR" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) | head -n 1)"
fi

if [ -z "$INPUT_IMAGE" ] || [ ! -f "$INPUT_IMAGE" ]; then
  echo "[ERROR] No valid input image found." >&2
  echo "[ERROR] Checked directory: $DEFAULT_IMAGE_DIR" >&2
  echo "[ERROR] Please run: bash scripts/prepare_coco_subset.sh 500" >&2
  exit 1
fi

cd "$MINDYOLO_DIR"

echo "[INFO] Running YOLOv8n prediction"
echo "[INFO] image_path=$INPUT_IMAGE"
echo "[INFO] device_target=$DEVICE_TARGET"

python demo/predict.py \
  --config ./configs/yolov8/yolov8n.yaml \
  --weight "$WEIGHT" \
  --image_path "$INPUT_IMAGE" \
  --device_target "$DEVICE_TARGET" \
  > "$LOG_FILE" 2>&1

if [ -d "$MINDYOLO_DIR/detect_results" ]; then
  cp -r "$MINDYOLO_DIR/detect_results/"* "$PROJECT_ROOT/results/detect_results/" 2>/dev/null || true
fi

echo "[OK] Prediction log: $LOG_FILE"
echo "[OK] Detection results: $PROJECT_ROOT/results/detect_results"
