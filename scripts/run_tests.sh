#!/usr/bin/env bash
# 本脚本统一执行语法检查和测试，并把结果写入 logs/tests.log。
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
{
  uv run python -m compileall -q src tests
  uv run pytest
} 2>&1 | tee logs/tests.log

