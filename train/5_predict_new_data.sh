#!/usr/bin/env bash
set -euo pipefail

TRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${TRAIN_DIR}/.." && pwd)"
PYTHON="${PYTHON:-/home/iaw/soft/conda/2024.06.1/envs/python3.12/bin/python}"

# Default production candidate selected from train.csv primary view:
#   selector split: seed=1, test_size=0.2
#   model: xgb
#   descriptor: rdkit_xtb_acsf
#   RMSE_test=0.2184, MAE_test=0.1514, R2_test=0.9427, Spearmanr_test=0.9395
# Override these with environment variables when predicting a different split/model.
TARGET="${TARGET:-ddg}"
HYPER_SPLIT="${HYPER_SPLIT:-seed_1_test_0-2}"
SEARCH_METHOD="${SEARCH_METHOD:-optuna}"
SPLIT="${SPLIT:-seed_1_test_0-2}"
MODEL_NAME="${MODEL_NAME:-xgb}"
DESCRIPTOR="${DESCRIPTOR:-rdkit_xtb_acsf}"
DATASET="${DATASET:-extra}"
SELECTION_SEED="${SELECTION_SEED:-1}"
SELECTION_TEST_SIZE="${SELECTION_TEST_SIZE:-0.2}"
SELECTION_RMSE_TEST="${SELECTION_RMSE_TEST:-0.2184}"
SELECTION_MAE_TEST="${SELECTION_MAE_TEST:-0.1514}"
SELECTION_R2_TEST="${SELECTION_R2_TEST:-0.9427}"
SELECTION_SPEARMANR_TEST="${SELECTION_SPEARMANR_TEST:-0.9395}"

MODEL_PATH="${MODEL_PATH:-${TRAIN_DIR}/output/pt/${TARGET}/hyper_${HYPER_SPLIT}/search_${SEARCH_METHOD}/${SPLIT}/${MODEL_NAME}_${DESCRIPTOR}.pkl}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/DataSet/Data_All/3_data_for_train/${TARGET}/${SPLIT}/${DESCRIPTOR}}"
OUTPUT="${OUTPUT:-${TRAIN_DIR}/predict_new/${TARGET}_hyper_${HYPER_SPLIT}_search_${SEARCH_METHOD}_split_${SPLIT}_${MODEL_NAME}_${DESCRIPTOR}_${DATASET}_predictions.csv}"
DATA_X="${DATA_X:-${DATA_DIR}/${DATASET}_data_x.npy}"

if [[ ! -f "${MODEL_PATH}" ]]; then
  printf '[FAIL] Model file does not exist: %s\n' "${MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -d "${DATA_DIR}" ]]; then
  printf '[FAIL] Data directory does not exist: %s\n' "${DATA_DIR}" >&2
  exit 1
fi
if [[ ! -f "${DATA_X}" ]]; then
  printf '[FAIL] Feature matrix does not exist: %s\n' "${DATA_X}" >&2
  exit 1
fi

printf '[INFO] AAReact prediction workflow\n'
printf '  python: %s\n' "${PYTHON}"
printf '  selected_by: seed=%s, test_size=%s\n' "${SELECTION_SEED}" "${SELECTION_TEST_SIZE}"
printf '  selected_metrics: RMSE_test=%s, MAE_test=%s, R2_test=%s, Spearmanr_test=%s\n' "${SELECTION_RMSE_TEST}" "${SELECTION_MAE_TEST}" "${SELECTION_R2_TEST}" "${SELECTION_SPEARMANR_TEST}"
printf '  target: %s\n' "${TARGET}"
printf '  hyper_split: %s\n' "${HYPER_SPLIT}"
printf '  search_method: %s\n' "${SEARCH_METHOD}"
printf '  split: %s\n' "${SPLIT}"
printf '  model: %s\n' "${MODEL_NAME}"
printf '  descriptor: %s\n' "${DESCRIPTOR}"
printf '  dataset: %s\n' "${DATASET}"
printf '  model_path: %s\n' "${MODEL_PATH}"
printf '  data_dir: %s\n' "${DATA_DIR}"
printf '  data_x: %s\n' "${DATA_X}"
printf '  output: %s\n' "${OUTPUT}"

cd "${PROJECT_ROOT}"
exec "${PYTHON}" "${TRAIN_DIR}/script/predict_new_data.py" \
  --target "${TARGET}" \
  --hyper-split "${HYPER_SPLIT}" \
  --search-method "${SEARCH_METHOD}" \
  --split "${SPLIT}" \
  --model-name "${MODEL_NAME}" \
  --descriptor "${DESCRIPTOR}" \
  --dataset "${DATASET}" \
  --model "${MODEL_PATH}" \
  --data-dir "${DATA_DIR}" \
  --data-x "${DATA_X}" \
  --output "${OUTPUT}" \
  "$@"
