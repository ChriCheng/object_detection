#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; mkdir -p "$ROOT/weights"; cd "$ROOT/weights"
URL="https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov8/yolov8-n_500e_mAP372-0e737186-910v2.ckpt"
[ -f yolov8n.ckpt ] || wget -c "$URL" -O yolov8n.ckpt
echo "[OK] weights/yolov8n.ckpt"
