#!/usr/bin/env bash

# ===== Default GPU settings =====
export DEVICE_TARGET="${DEVICE_TARGET:-GPU}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ===== CUDA path =====
# 优先使用项目已验证可用的 CUDA 12.8
if [ -d "/usr/local/cuda-12.8" ]; then
  export CUDA_HOME=/usr/local/cuda-12.8
elif [ -d "/usr/local/cuda-12.1" ]; then
  export CUDA_HOME=/usr/local/cuda-12.1
elif [ -d "/usr/local/cuda-11.8" ]; then
  export CUDA_HOME=/usr/local/cuda-11.8
elif [ -d "/usr/local/cuda" ]; then
  export CUDA_HOME=/usr/local/cuda
fi

# 避免 LD_LIBRARY_PATH 为空时报错
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

# CUDA 动态库
if [ -n "${CUDA_HOME:-}" ]; then
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
fi

# conda 环境动态库，例如 cudnn / cublas / nvrtc
if [ -n "${CONDA_PREFIX:-}" ]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
fi

# NVIDIA 驱动库常见位置
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

echo "[MindSpore GPU ENV] DEVICE_TARGET=${DEVICE_TARGET}"
echo "[MindSpore GPU ENV] CUDA_HOME=${CUDA_HOME:-not_set}"
echo "[MindSpore GPU ENV] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[MindSpore GPU ENV] LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
