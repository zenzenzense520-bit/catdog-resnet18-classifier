#!/usr/bin/env bash
# 本脚本统一启动单图推理并把输出写入 logs/predict-command.log。
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
uv run catdog-predict "$@" 2>&1 | tee logs/predict-command.log

