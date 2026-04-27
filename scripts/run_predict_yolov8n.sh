#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; M="$ROOT/third_party/mindyolo"; DEV="${DEVICE_TARGET:-CPU}"; W="$ROOT/weights/yolov8n.ckpt"; IMG="${IMAGE_PATH:-$ROOT/data/coco_mindyolo/coco/images/val2017}"
mkdir -p "$ROOT/logs" "$ROOT/results/detect_results"
[ -d "$M" ] || { echo "Run scripts/setup_mindyolo.sh first"; exit 1; }
[ -f "$W" ] || { echo "Run scripts/download_yolov8n_weight.sh first"; exit 1; }
cd "$M"
python demo/predict.py --config ./configs/yolov8/yolov8n.yaml --weight "$W" --image_path "$IMG" --device_target "$DEV" > "$ROOT/logs/predict_yolov8n.log" 2>&1
[ -d "$M/detect_results" ] && cp -r "$M/detect_results"/* "$ROOT/results/detect_results/" 2>/dev/null || true
