#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 默认启用 MindSpore GPU 环境
source "$PROJECT_ROOT/scripts/env_mindspore_gpu.sh"

MINDYOLO_DIR="$PROJECT_ROOT/third_party/mindyolo"
DEVICE_TARGET="${DEVICE_TARGET:-GPU}"
WEIGHT="$PROJECT_ROOT/weights/yolov8n.ckpt"

LOG_FILE="$PROJECT_ROOT/logs/eval_yolov8n.log"
METRIC_JSON="$PROJECT_ROOT/results/metrics/yolov8n_eval_result.json"
EVAL_RESULT_DIR="$PROJECT_ROOT/results/runs_test"

COCO_ROOT="$PROJECT_ROOT/data/coco_mindyolo/coco"
VAL_TXT="$COCO_ROOT/val2017.txt"

mkdir -p "$PROJECT_ROOT/logs" "$PROJECT_ROOT/results/metrics" "$EVAL_RESULT_DIR"

if [ ! -d "$MINDYOLO_DIR" ]; then
  echo "[ERROR] MindYOLO not found. Run: bash scripts/setup_mindyolo.sh" >&2
  exit 1
fi

if [ ! -f "$WEIGHT" ]; then
  echo "[ERROR] Weight not found: $WEIGHT. Run: bash scripts/download_yolov8n_weight.sh" >&2
  exit 1
fi

if [ ! -f "$VAL_TXT" ]; then
  echo "[ERROR] COCO subset not prepared. Run: bash scripts/prepare_coco_subset.sh 500" >&2
  exit 1
fi

# 防止 val2017.txt 又被旧脚本生成成 ./coco/images/...
sed -i 's#^\./coco/images/#./images/#' "$VAL_TXT"

FIRST_LINE="$(head -n 1 "$VAL_TXT")"
FIRST_IMAGE="$COCO_ROOT/${FIRST_LINE#./}"

echo "[INFO] VAL_TXT first line: $FIRST_LINE"
echo "[INFO] First image check: $FIRST_IMAGE"

if [ ! -f "$FIRST_IMAGE" ]; then
  echo "[ERROR] First image does not exist: $FIRST_IMAGE" >&2
  exit 1
fi

# 确保 MindYOLO 工作目录下存在 coco 符号链接
rm -rf "$MINDYOLO_DIR/coco"
ln -s "$COCO_ROOT" "$MINDYOLO_DIR/coco"

# 每次评估前删除旧缓存，避免 val2017.txt 改过但 cache 仍然沿用旧路径
rm -f "$COCO_ROOT/val2017.cache"*
rm -f "$MINDYOLO_DIR/coco/val2017.cache"*

cd "$MINDYOLO_DIR"

echo "[INFO] Running YOLOv8n evaluation"
echo "[INFO] DEVICE_TARGET=$DEVICE_TARGET"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}"
echo "[INFO] WEIGHT=$WEIGHT"
echo "[INFO] save_dir=$EVAL_RESULT_DIR"

python test.py \
  --config ./configs/yolov8/yolov8n.yaml \
  --weight "$WEIGHT" \
  --device_target "$DEVICE_TARGET" \
  --save_dir "$EVAL_RESULT_DIR" \
  > "$LOG_FILE" 2>&1

python "$PROJECT_ROOT/src/parse_mindyolo_eval_log.py" \
  --log_file "$LOG_FILE" \
  --out_file "$METRIC_JSON" || true

# 复制一份日志到 results，方便最终提交
cp "$LOG_FILE" "$PROJECT_ROOT/results/metrics/eval_yolov8n.log"

echo "[OK] Eval log: $LOG_FILE"
echo "[OK] Eval log copy: $PROJECT_ROOT/results/metrics/eval_yolov8n.log"
echo "[OK] Parsed metrics: $METRIC_JSON"
echo "[OK] Raw eval result dir: $EVAL_RESULT_DIR"
