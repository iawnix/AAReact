#!/usr/bin/env bash
set -euo pipefail

TRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${TRAIN_DIR}/.." && pwd)"
PYTHON="${PYTHON:-/home/iaw/soft/conda/2024.06.1/envs/python3.12/bin/python}"

# Default production candidate selected from train.csv primary view:
#   selector split: seed=1, test_size=0.2
#   model: xgb
#   descriptor: rdkit_xtb_acsf
TARGET="${TARGET:-ddg}"
HYPER_SPLIT="${HYPER_SPLIT:-seed_1_test_0-2}"
SEARCH_METHOD="${SEARCH_METHOD:-optuna}"
SPLIT="${SPLIT:-seed_1_test_0-2}"
MODEL_NAME="${MODEL_NAME:-xgb}"
DESCRIPTOR="${DESCRIPTOR:-rdkit_xtb_acsf}"
PRIMARY_SEED="${PRIMARY_SEED:-1}"
PRIMARY_TEST_SIZE="${PRIMARY_TEST_SIZE:-0.2}"
MAX_SHAP_SAMPLES="${MAX_SHAP_SAMPLES:-0}"
MAX_DISPLAY="${MAX_DISPLAY:-20}"
OUTPUT_DIR="${OUTPUT_DIR:-${TRAIN_DIR}/Analysis/selected_model}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/config/train/${TARGET}/hyper_${HYPER_SPLIT}/search_${SEARCH_METHOD}/${SPLIT}/train_ml_${MODEL_NAME}_${DESCRIPTOR}.toml}"
MODEL_PATH="${MODEL_PATH:-${TRAIN_DIR}/output/pt/${TARGET}/hyper_${HYPER_SPLIT}/search_${SEARCH_METHOD}/${SPLIT}/${MODEL_NAME}_${DESCRIPTOR}.pkl}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/DataSet/Data_All/3_data_for_train/${TARGET}/${SPLIT}/${DESCRIPTOR}}"
SOURCE_CSV="${SOURCE_CSV:-}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-${TRAIN_DIR}/.mplconfig}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${TRAIN_DIR}/.cache}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}" "${OUTPUT_DIR}"

info() { printf '\033[1;36m[INFO]\033[0m %s\n' "$*"; }
ok() { printf '\033[1;32m[OK]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[FAIL]\033[0m %s\n' "$*" >&2; }

if [[ ! -f "${CONFIG}" ]]; then
  fail "Train config does not exist: ${CONFIG}"
  exit 1
fi
if [[ ! -f "${MODEL_PATH}" ]]; then
  fail "Model file does not exist: ${MODEL_PATH}"
  exit 1
fi
if [[ ! -d "${DATA_DIR}" ]]; then
  fail "Data directory does not exist: ${DATA_DIR}"
  exit 1
fi

info "AAReact selected-model analysis"
printf '  python: %s\n' "${PYTHON}"
printf '  target: %s\n' "${TARGET}"
printf '  hyper_split: %s\n' "${HYPER_SPLIT}"
printf '  search_method: %s\n' "${SEARCH_METHOD}"
printf '  split: %s\n' "${SPLIT}"
printf '  model: %s\n' "${MODEL_NAME}"
printf '  descriptor: %s\n' "${DESCRIPTOR}"
printf '  primary_seed: %s\n' "${PRIMARY_SEED}"
printf '  primary_test_size: %s\n' "${PRIMARY_TEST_SIZE}"
printf '  config: %s\n' "${CONFIG}"
printf '  model_path: %s\n' "${MODEL_PATH}"
printf '  data_dir: %s\n' "${DATA_DIR}"
printf '  output_dir: %s\n' "${OUTPUT_DIR}"
if [[ -n "${SOURCE_CSV}" ]]; then
  printf '  source_csv: %s\n' "${SOURCE_CSV}"
fi

SOURCE_ARGS=()
if [[ -n "${SOURCE_CSV}" ]]; then
  SOURCE_ARGS=(--source-csv "${SOURCE_CSV}")
fi

cd "${PROJECT_ROOT}"
"${PYTHON}" "${TRAIN_DIR}/script/analyze_selected_model.py" \
  --target "${TARGET}" \
  --hyper-split "${HYPER_SPLIT}" \
  --search-method "${SEARCH_METHOD}" \
  --split "${SPLIT}" \
  --model-name "${MODEL_NAME}" \
  --descriptor "${DESCRIPTOR}" \
  --config "${CONFIG}" \
  --model "${MODEL_PATH}" \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --primary-seed "${PRIMARY_SEED}" \
  --primary-test-size "${PRIMARY_TEST_SIZE}" \
  --max-shap-samples "${MAX_SHAP_SAMPLES}" \
  --max-display "${MAX_DISPLAY}" \
  "${SOURCE_ARGS[@]}" \
  "$@"

ok "Selected-model analysis finished: ${OUTPUT_DIR}"
