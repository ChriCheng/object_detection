#!/usr/bin/env bash

# ===== MindSpore GPU environment =====
# 根据你服务器实际 CUDA 版本选择路径。
# 如果 /usr/local/cuda 指向 CUDA 13，而 MindSpore 不支持，建议改成 /usr/local/cuda-12.8 或实际可用版本。

if [ -d "/usr/local/cuda-12.8" ]; then
  export CUDA_HOME=/usr/local/cuda-12.8
elif [ -d "/usr/local/cuda-12.1" ]; then
  export CUDA_HOME=/usr/local/cuda-12.1
elif [ -d "/usr/local/cuda-11.8" ]; then
  export CUDA_HOME=/usr/local/cuda-11.8
elif [ -d "/usr/local/cuda" ]; then
  export CUDA_HOME=/usr/local/cuda
fi

if [ -n "${CUDA_HOME:-}" ]; then
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
fi

# conda 环境里如果装了 cudnn/cublas/nvrtc，优先加入 conda lib
if [ -n "${CONDA_PREFIX:-}" ]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
fi

# 系统 NVIDIA 驱动库常见位置
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

# 指定用第 0 张 GPU
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "[MindSpore GPU ENV] CUDA_HOME=${CUDA_HOME:-not_set}"
echo "[MindSpore GPU ENV] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[MindSpore GPU ENV] LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
