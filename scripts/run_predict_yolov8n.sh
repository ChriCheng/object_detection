#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 默认启用 MindSpore GPU 环境
source "$PROJECT_ROOT/scripts/env_mindspore_gpu.sh"

MINDYOLO_DIR="$PROJECT_ROOT/third_party/mindyolo"
DEVICE_TARGET="${DEVICE_TARGET:-GPU}"
WEIGHT="$PROJECT_ROOT/weights/yolov8n.ckpt"

# 单独划分出来的可视化测试集目录
PREDICT_IMAGE_DIR="${PREDICT_IMAGE_DIR:-$PROJECT_ROOT/data/predict_images}"

LOG_FILE="$PROJECT_ROOT/logs/predict_yolov8n.log"
PRED_RESULT_DIR="$PROJECT_ROOT/results/detect_results"
PRED_RAW_DIR="$PROJECT_ROOT/results/runs_infer"

# 默认推断 10 张图
PREDICT_NUM="${PREDICT_NUM:-10}"

mkdir -p "$PROJECT_ROOT/logs" "$PRED_RESULT_DIR" "$PRED_RAW_DIR"

if [ ! -d "$MINDYOLO_DIR" ]; then
  echo "[ERROR] MindYOLO not found. Run: bash scripts/setup_mindyolo.sh" >&2
  exit 1
fi

if [ ! -f "$WEIGHT" ]; then
  echo "[ERROR] Weight not found: $WEIGHT. Run: bash scripts/download_yolov8n_weight.sh" >&2
  exit 1
fi

# 如果 data/predict_images 不存在或为空，就从 val2017 里自动复制 PREDICT_NUM 张
if [ ! -d "$PREDICT_IMAGE_DIR" ] || [ -z "$(find "$PREDICT_IMAGE_DIR" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | head -n 1)" ]; then
  echo "[INFO] Predict image directory not found or empty. Creating it from val2017..."
  mkdir -p "$PREDICT_IMAGE_DIR"

  find "$PROJECT_ROOT/data/coco_mindyolo/coco/images/val2017" \
    -maxdepth 1 \
    -type f \
    \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) \
    | sort \
    | head -n "$PREDICT_NUM" \
    | xargs -I{} cp {} "$PREDICT_IMAGE_DIR/"
fi

# 清空旧可视化结果，避免混入历史图片
rm -rf "$PRED_RESULT_DIR" "$PRED_RAW_DIR"
mkdir -p "$PRED_RESULT_DIR" "$PRED_RAW_DIR"

cd "$MINDYOLO_DIR"

echo "[INFO] Running YOLOv8n prediction"
echo "[INFO] DEVICE_TARGET=$DEVICE_TARGET"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}"
echo "[INFO] PREDICT_IMAGE_DIR=$PREDICT_IMAGE_DIR"
echo "[INFO] PREDICT_NUM=$PREDICT_NUM"
echo "[INFO] Final results dir=$PRED_RESULT_DIR"

: > "$LOG_FILE"

# 如果用户指定 IMAGE_PATH，只推断这一张；否则从 PREDICT_IMAGE_DIR 取前 PREDICT_NUM 张
if [ -n "${IMAGE_PATH:-}" ]; then
  if [[ "$IMAGE_PATH" = /* ]]; then
    IMAGE_LIST="$IMAGE_PATH"
  else
    IMAGE_LIST="$PROJECT_ROOT/$IMAGE_PATH"
  fi
else
  IMAGE_LIST="$(find "$PREDICT_IMAGE_DIR" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | sort | head -n "$PREDICT_NUM")"
fi

if [ -z "$IMAGE_LIST" ]; then
  echo "[ERROR] No input image found in $PREDICT_IMAGE_DIR." >&2
  exit 1
fi

idx=0
while IFS= read -r INPUT_IMAGE; do
  idx=$((idx + 1))

  if [ -z "$INPUT_IMAGE" ]; then
    continue
  fi

  if [ ! -f "$INPUT_IMAGE" ]; then
    echo "[WARN] Skip missing image: $INPUT_IMAGE" | tee -a "$LOG_FILE"
    continue
  fi

  BASENAME="$(basename "$INPUT_IMAGE")"
  STEM="${BASENAME%.*}"

  echo "[INFO] [$idx] Predict image: $INPUT_IMAGE" | tee -a "$LOG_FILE"

  # 清理 MindYOLO 默认输出，避免复制到上一张图片
  rm -rf "$MINDYOLO_DIR/detect_results"

  # 创建时间标记：用于只收集本次 predict 新生成的图片
  MARKER_FILE="$PROJECT_ROOT/results/.predict_marker_${idx}"
  touch "$MARKER_FILE"

  # 优先尝试 --save_dir，把原始结果保存到 results/runs_infer
  # 如果当前 MindYOLO 版本不支持 --save_dir，则退回默认输出目录
  if python demo/predict.py -h 2>&1 | grep -q -- "--save_dir"; then
    python demo/predict.py \
      --config ./configs/yolov8/yolov8n.yaml \
      --weight "$WEIGHT" \
      --image_path "$INPUT_IMAGE" \
      --device_target "$DEVICE_TARGET" \
      --save_dir "$PRED_RAW_DIR" \
      >> "$LOG_FILE" 2>&1
  else
    python demo/predict.py \
      --config ./configs/yolov8/yolov8n.yaml \
      --weight "$WEIGHT" \
      --image_path "$INPUT_IMAGE" \
      --device_target "$DEVICE_TARGET" \
      >> "$LOG_FILE" 2>&1
  fi

  FOUND=0

  # 兜底搜索可能输出位置：
  # 1. results/runs_infer
  # 2. results/detect_results
  # 3. third_party/mindyolo/detect_results
  # 4. third_party/mindyolo/runs_infer
  while IFS= read -r OUT_IMG; do
    [ -z "$OUT_IMG" ] && continue
    EXT="${OUT_IMG##*.}"
    cp "$OUT_IMG" "$PRED_RESULT_DIR/${idx}_${STEM}_pred.${EXT}"
    FOUND=1
  done < <(
    find "$PRED_RAW_DIR" "$PRED_RESULT_DIR" "$MINDYOLO_DIR/detect_results" "$MINDYOLO_DIR/runs_infer" \
      -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) \
      -newer "$MARKER_FILE" 2>/dev/null | sort
  )

  rm -f "$MARKER_FILE"

  if [ "$FOUND" -eq 0 ]; then
    echo "[WARN] No prediction image collected for $INPUT_IMAGE" | tee -a "$LOG_FILE"
    echo "[WARN] Checking possible output dirs:" | tee -a "$LOG_FILE"
    find "$PRED_RAW_DIR" "$MINDYOLO_DIR/detect_results" "$MINDYOLO_DIR/runs_infer" \
      -maxdepth 4 \
      -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) \
      2>/dev/null | head -n 20 | tee -a "$LOG_FILE" || true
  fi

done <<< "$IMAGE_LIST"

echo "[OK] Prediction log: $LOG_FILE"
echo "[OK] Final detect results: $PRED_RESULT_DIR"
echo "[INFO] Saved images:"
find "$PRED_RESULT_DIR" -maxdepth 1 -type f | sort | head -n 50
