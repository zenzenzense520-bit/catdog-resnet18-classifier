#!/usr/bin/env bash
# 本脚本从 Microsoft 官方镜像下载并解压 Kaggle Cats vs Dogs 数据集。
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data logs
archive="data/kagglecatsanddogs_5340.zip"
url="https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
{
  if [[ ! -f "$archive" ]]; then
    curl --fail --location --retry 3 --output "$archive" "$url"
  fi
  if [[ ! -d data/PetImages/Cat || ! -d data/PetImages/Dog ]]; then
    unzip -q "$archive" -d data
  fi
  printf 'Cat=%s Dog=%s\n' \
    "$(find data/PetImages/Cat -type f | wc -l)" \
    "$(find data/PetImages/Dog -type f | wc -l)"
} 2>&1 | tee logs/download-data.log
