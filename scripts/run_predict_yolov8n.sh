#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 默认启用 MindSpore GPU 环境
source "$PROJECT_ROOT/scripts/env_mindspore_gpu.sh"

MINDYOLO_DIR="$PROJECT_ROOT/third_party/mindyolo"
DEVICE_TARGET="${DEVICE_TARGET:-GPU}"
WEIGHT="$PROJECT_ROOT/weights/yolov8n.ckpt"
DEFAULT_IMAGE_DIR="$PROJECT_ROOT/data/coco_mindyolo/coco/images/val2017"

LOG_FILE="$PROJECT_ROOT/logs/predict_yolov8n.log"
PRED_RESULT_DIR="$PROJECT_ROOT/results/detect_results"
PRED_RAW_DIR="$PROJECT_ROOT/results/runs_infer"

mkdir -p "$PROJECT_ROOT/logs" "$PRED_RESULT_DIR" "$PRED_RAW_DIR"

if [ ! -d "$MINDYOLO_DIR" ]; then
  echo "[ERROR] MindYOLO not found. Run: bash scripts/setup_mindyolo.sh" >&2
  exit 1
fi

if [ ! -f "$WEIGHT" ]; then
  echo "[ERROR] Weight not found: $WEIGHT. Run: bash scripts/download_yolov8n_weight.sh" >&2
  exit 1
fi

if [ ! -d "$DEFAULT_IMAGE_DIR" ]; then
  echo "[ERROR] Image directory not found: $DEFAULT_IMAGE_DIR" >&2
  echo "[ERROR] Please run: bash scripts/prepare_coco_subset.sh 500" >&2
  exit 1
fi

# demo/predict.py 需要单张图片文件
if [ -n "${IMAGE_PATH:-}" ]; then
  if [[ "$IMAGE_PATH" = /* ]]; then
    INPUT_IMAGE="$IMAGE_PATH"
  else
    INPUT_IMAGE="$PROJECT_ROOT/$IMAGE_PATH"
  fi
else
  INPUT_IMAGE="$(find "$DEFAULT_IMAGE_DIR" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | sort | head -n 1)"
fi

if [ -z "$INPUT_IMAGE" ] || [ ! -f "$INPUT_IMAGE" ]; then
  echo "[ERROR] Detect: input image file not available." >&2
  echo "[ERROR] INPUT_IMAGE=$INPUT_IMAGE" >&2
  echo "[ERROR] DEFAULT_IMAGE_DIR=$DEFAULT_IMAGE_DIR" >&2
  exit 1
fi

cd "$MINDYOLO_DIR"

echo "[INFO] Running YOLOv8n prediction"
echo "[INFO] DEVICE_TARGET=$DEVICE_TARGET"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}"
echo "[INFO] image_path=$INPUT_IMAGE"
echo "[INFO] project result dir=$PRED_RESULT_DIR"

# 清理 MindYOLO 默认旧输出，避免复制到历史图片
rm -rf "$MINDYOLO_DIR/detect_results"

# 如果当前 MindYOLO predict.py 支持 --save_dir，就直接保存到 results/runs_infer；
# 如果不支持，则仍按官方默认输出到 ./detect_results，后面再复制。
if python demo/predict.py -h 2>&1 | grep -q -- "--save_dir"; then
  python demo/predict.py \
    --config ./configs/yolov8/yolov8n.yaml \
    --weight "$WEIGHT" \
    --image_path "$INPUT_IMAGE" \
    --device_target "$DEVICE_TARGET" \
    --save_dir "$PRED_RAW_DIR" \
    > "$LOG_FILE" 2>&1
else
  python demo/predict.py \
    --config ./configs/yolov8/yolov8n.yaml \
    --weight "$WEIGHT" \
    --image_path "$INPUT_IMAGE" \
    --device_target "$DEVICE_TARGET" \
    > "$LOG_FILE" 2>&1
fi

# 兜底同步：把 MindYOLO 可能生成的图片都整理到 results/detect_results
find "$MINDYOLO_DIR/detect_results" "$PRED_RAW_DIR" \
  -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) \
  -exec cp {} "$PRED_RESULT_DIR/" \; 2>/dev/null || true

echo "[OK] Prediction log: $LOG_FILE"
echo "[OK] Raw inference dir: $PRED_RAW_DIR"
echo "[OK] Final detect results: $PRED_RESULT_DIR"

echo "[INFO] Current files in results/detect_results:"
find "$PRED_RESULT_DIR" -maxdepth 1 -type f | head -n 20
