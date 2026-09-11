#!/usr/bin/env bash
# 本脚本统一创建 .venv 并安装已批准的固定版本依赖。
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
uv sync 2>&1 | tee logs/setup.log

