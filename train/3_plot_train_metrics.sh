#!/usr/bin/env bash
set -euo pipefail

TRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${TRAIN_DIR}/.." && pwd)"
PYTHON="${PYTHON:-/home/iaw/soft/conda/2024.06.1/envs/python3.12/bin/python}"
PRIMARY_SEED="${PRIMARY_SEED:-1}"
PRIMARY_TEST_SIZE="${PRIMARY_TEST_SIZE:-0.2}"

cd "${PROJECT_ROOT}"
exec "${PYTHON}" "${TRAIN_DIR}/script/plot_train_metrics.py" \
  --primary-seed "${PRIMARY_SEED}" \
  --primary-test-size "${PRIMARY_TEST_SIZE}" \
  "$@"
