#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/iaw/DATA2/AAReact"
PYTHON="${PYTHON:-/home/iaw/soft/conda/2024.06.1/envs/python3.12/bin/python}"
RUN_NAME="rea1_train_rea234_test"
DATA_BASE="${PROJECT_ROOT}/DataSet/Data_All3"
DATA_DIR="${DATA_BASE}/3_data_for_train/ddg/${RUN_NAME}/rdkit_xtb_acsf"
OUTPUT_DIR="${PROJECT_ROOT}/train/output/Data_All3/rea1_train_rea234_test"
STUDY_DIR="${OUTPUT_DIR}/optuna/study"

mkdir -p "${DATA_BASE}" "${OUTPUT_DIR}"

"${PYTHON}" "${PROJECT_ROOT}/DataSet/scripts/build_rea_external_dataset.py" \
  --base "${DATA_BASE}" \
  --source-csv "${PROJECT_ROOT}/DataSet/Data_All/full_data_436-20260723.csv" \
  --raw-feature-dir "${PROJECT_ROOT}/DataSet/Data_All/2_raw_features" \
  --train-rea REA-1 \
  --test-rea REA-2 REA-3 REA-4 \
  --batch 0 \
  --target ddg \
  --run-name "${RUN_NAME}"

"${PYTHON}" "${PROJECT_ROOT}/train/script/run_rea_external_experiment.py" \
  --data-dir "${DATA_DIR}" \
  --source-csv "${DATA_BASE}/full_data_rea1_train_rea234_test-20260922.csv" \
  --output-dir "${OUTPUT_DIR}" \
  --target ddg \
  --seed 1 \
  --cv 5 \
  --n-trials 400 \
  --n-startup-trials 20 \
  --n-cpu 5
