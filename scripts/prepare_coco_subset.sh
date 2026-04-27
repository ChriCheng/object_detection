#!/usr/bin/env bash
set -euo pipefail
N="${1:-500}"; ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python "$ROOT/src/prepare_coco_subset.py" --image_dir "$ROOT/data/raw/coco/val2017" --ann_file "$ROOT/data/raw/coco/annotations/instances_val2017.json" --out_root "$ROOT/data/coco_mindyolo/coco" --num_images "$N" --copy_mode copy
[ -d "$ROOT/third_party/mindyolo" ] && { rm -rf "$ROOT/third_party/mindyolo/coco"; ln -s "$ROOT/data/coco_mindyolo/coco" "$ROOT/third_party/mindyolo/coco"; }
