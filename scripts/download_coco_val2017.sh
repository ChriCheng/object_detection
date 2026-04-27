#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
D="$ROOT/data/raw/coco"; mkdir -p "$D"; cd "$D"
[ -d val2017 ] || { [ -f val2017.zip ] || wget -c http://images.cocodataset.org/zips/val2017.zip; unzip -q val2017.zip; }
[ -f annotations/instances_val2017.json ] || { [ -f annotations_trainval2017.zip ] || wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip; unzip -q annotations_trainval2017.zip; }
echo "[OK] COCO val2017 ready"
