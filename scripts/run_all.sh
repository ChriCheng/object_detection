#!/usr/bin/env bash
set -euo pipefail

NUM_IMAGES="${1:-500}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 默认启用 MindSpore GPU 环境
source "$PROJECT_ROOT/scripts/env_mindspore_gpu.sh"

bash "$PROJECT_ROOT/scripts/setup_mindyolo.sh"
bash "$PROJECT_ROOT/scripts/download_yolov8n_weight.sh"

if [ ! -d "$PROJECT_ROOT/data/raw/coco/val2017" ] || [ ! -f "$PROJECT_ROOT/data/raw/coco/annotations/instances_val2017.json" ]; then
  bash "$PROJECT_ROOT/scripts/download_coco_val2017.sh"
fi

bash "$PROJECT_ROOT/scripts/prepare_coco_subset.sh" "$NUM_IMAGES"
bash "$PROJECT_ROOT/scripts/run_predict_yolov8n.sh"
bash "$PROJECT_ROOT/scripts/run_eval_yolov8n.sh"

python "$PROJECT_ROOT/src/summarize_project.py" || true
