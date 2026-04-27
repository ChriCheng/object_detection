#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
M="$ROOT/third_party/mindyolo"
mkdir -p "$ROOT/third_party"
[ -d "$M/.git" ] || git clone https://github.com/mindspore-lab/mindyolo.git "$M"
cd "$M"
git rev-parse v0.5.0 >/dev/null 2>&1 && git checkout v0.5.0 || true
pip install -r requirements.txt
pip install -e .
rm -rf "$M/coco" && ln -s "$ROOT/data/coco_mindyolo/coco" "$M/coco"
echo "[OK] MindYOLO ready"
