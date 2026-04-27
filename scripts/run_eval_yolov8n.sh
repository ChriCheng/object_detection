#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ "${DEVICE_TARGET:-CPU}" = "GPU" ]; then
  source "$PROJECT_ROOT/scripts/env_mindspore_gpu.sh"
fi

MINDYOLO_DIR="$PROJECT_ROOT/third_party/mindyolo"
DEVICE_TARGET="${DEVICE_TARGET:-CPU}"
WEIGHT="$PROJECT_ROOT/weights/yolov8n.ckpt"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; M="$ROOT/third_party/mindyolo"; DEV="${DEVICE_TARGET:-CPU}"; W="$ROOT/weights/yolov8n.ckpt"
mkdir -p "$ROOT/logs" "$ROOT/results/metrics"
[ -d "$M" ] || { echo "Run scripts/setup_mindyolo.sh first"; exit 1; }
[ -f "$W" ] || { echo "Run scripts/download_yolov8n_weight.sh first"; exit 1; }
rm -rf "$M/coco" && ln -s "$ROOT/data/coco_mindyolo/coco" "$M/coco"
cd "$M"
python test.py --config ./configs/yolov8/yolov8n.yaml --weight "$W" --device_target "$DEV" > "$ROOT/logs/eval_yolov8n.log" 2>&1
python "$ROOT/src/parse_mindyolo_eval_log.py" --log_file "$ROOT/logs/eval_yolov8n.log" --out_file "$ROOT/results/metrics/yolov8n_eval_result.json" || true
