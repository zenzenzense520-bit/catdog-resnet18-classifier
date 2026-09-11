#!/usr/bin/env bash
# 本脚本统一启动训练并把完整输出写入 logs/train-command.log。
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
uv run catdog-train "$@" 2>&1 | tee logs/train-command.log

