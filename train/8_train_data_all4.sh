#!/usr/bin/env bash
set -euo pipefail

TRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${TRAIN_DIR}/.." && pwd)"
PYTHON="${PYTHON:-/home/iaw/soft/conda/2024.06.1/envs/python3.12/bin/python}"

TARGET="${TARGET:-ddg}"
SPLIT="${SPLIT:-seed_1_test_0-2}"
MODEL="${MODEL:-xgb}"
DESCRIPTOR="${DESCRIPTOR:-rdkit_xtb_acsf}"
HYPER_SPLIT="${HYPER_SPLIT:-seed_1_test_0-2}"
SEARCH_METHOD="${SEARCH_METHOD:-optuna}"
OUTPUT_DIR="${OUTPUT_DIR:-${TRAIN_DIR}/output/Data_All4/${MODEL}_${DESCRIPTOR}}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-${TRAIN_DIR}/.mplconfig}"
mkdir -p "${MPLCONFIGDIR}"

exec "${PYTHON}" "${TRAIN_DIR}/script/run_data_all4_experiment.py" \
  --dataset-root "${PROJECT_ROOT}/DataSet/Data_All4" \
  --target "${TARGET}" \
  --split "${SPLIT}" \
  --model "${MODEL}" \
  --descriptor "${DESCRIPTOR}" \
  --hyper-split "${HYPER_SPLIT}" \
  --search-method "${SEARCH_METHOD}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
