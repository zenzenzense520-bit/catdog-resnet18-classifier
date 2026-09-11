#!/usr/bin/env bash
# 本脚本加载 ImageNet 预训练权重，并检查 CUDA 训练能力。
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
uv run python -c "import torch; from catdog_classifier.model import build_resnet18; model = build_resnet18(2, pretrained=True); print(f'model={type(model).__name__} torch={torch.__version__} cuda_build={torch.version.cuda} cuda_available={torch.cuda.is_available()}')" 2>&1 | tee logs/check-model.log
