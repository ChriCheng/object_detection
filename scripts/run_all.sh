#!/usr/bin/env bash
set -euo pipefail
N="${1:-500}"; ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bash "$ROOT/scripts/setup_mindyolo.sh"
bash "$ROOT/scripts/download_yolov8n_weight.sh"
[ -d "$ROOT/data/raw/coco/val2017" ] && [ -f "$ROOT/data/raw/coco/annotations/instances_val2017.json" ] || bash "$ROOT/scripts/download_coco_val2017.sh"
bash "$ROOT/scripts/prepare_coco_subset.sh" "$N"
bash "$ROOT/scripts/run_predict_yolov8n.sh"
bash "$ROOT/scripts/run_eval_yolov8n.sh"
python "$ROOT/src/summarize_project.py"
