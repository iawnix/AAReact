#!/usr/bin/env bash
set -euo pipefail

DATA_ALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${DATA_ALL_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-/home/iaw/soft/conda/2024.06.1/envs/python3.12/bin/python}"
LOG_DIR="${LOG_DIR:-${DATA_ALL_DIR}/logs}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
SOURCE_CSV="${SOURCE_CSV:-${DATA_ALL_DIR}/full_data_436-20260723.csv}"

SEEDS=(1 12 42)
TEST_SIZES=(0.15 0.2 0.3)
TRAIN_BATCHES="${TRAIN_BATCHES:-0}"
EXTRA_BATCHES="${EXTRA_BATCHES:-1}"
TARGET="ddg"

if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  BOLD="$(printf '\033[1m')"
  DIM="$(printf '\033[2m')"
  RED="$(printf '\033[31m')"
  GREEN="$(printf '\033[32m')"
  YELLOW="$(printf '\033[33m')"
  BLUE="$(printf '\033[34m')"
  RESET="$(printf '\033[0m')"
else
  BOLD=""
  DIM=""
  RED=""
  GREEN=""
  YELLOW=""
  BLUE=""
  RESET=""
fi

info() {
  printf "%b\n" "${BLUE}${BOLD}[INFO]${RESET} $*"
}

ok() {
  printf "%b\n" "${GREEN}${BOLD}[OK]${RESET} $*"
}

warn() {
  printf "%b\n" "${YELLOW}${BOLD}[WARN]${RESET} $*"
}

fail() {
  printf "%b\n" "${RED}${BOLD}[FAIL]${RESET} $*" >&2
}

run_logged() {
  local label="$1"
  local log_file="$2"
  shift 2

  info "${label}"
  {
    echo
    echo "===== ${label} ====="
    echo "command: $*"
    echo "started_at: $(date --iso-8601=seconds)"
  } >> "${log_file}"

  if "$@" >> "${log_file}" 2>&1; then
    echo "finished_at: $(date --iso-8601=seconds)" >> "${log_file}"
    ok "${label}"
  else
    fail "${label}"
    warn "Detailed log: ${log_file}"
    tail -80 "${log_file}" >&2 || true
    exit 1
  fi
}

mkdir -p "${LOG_DIR}"

info "AAReact Data_All ML dataset workflow"
printf "  python: %s\n" "${PYTHON}"
printf "  seeds: %s\n" "${SEEDS[*]}"
printf "  test_sizes: %s\n" "${TEST_SIZES[*]}"
printf "  train_batches: %s\n" "${TRAIN_BATCHES}"
printf "  extra_batches: %s\n" "${EXTRA_BATCHES}"
printf "  target: %s\n" "${TARGET}"
printf "  source_csv: %s\n" "${SOURCE_CSV}"
printf "  log_dir: %s\n" "${LOG_DIR}"
printf "  run_id: %s\n" "${RUN_ID}"

if [ ! -s "${SOURCE_CSV}" ]; then
  fail "Missing source CSV: ${SOURCE_CSV}"
  exit 1
fi
ok "Source CSV is present."

for feature_csv in \
  "${DATA_ALL_DIR}/2_raw_features/rdkit_desc_features.csv" \
  "${DATA_ALL_DIR}/2_raw_features/xtb_features.csv" \
  "${DATA_ALL_DIR}/2_raw_features/soap_features.csv" \
  "${DATA_ALL_DIR}/2_raw_features/acsf_features.csv"; do
  if [ ! -s "${feature_csv}" ]; then
    fail "Missing raw feature file: ${feature_csv}"
    warn "Run ${DATA_ALL_DIR}/1_build_raw_features.sh before this workflow."
    exit 1
  fi
done
ok "Raw feature files are present."

for seed in "${SEEDS[@]}"; do
  for test_size in "${TEST_SIZES[@]}"; do
    split_name="seed_${seed}_test_${test_size//./-}"
    split_log="${LOG_DIR}/${RUN_ID}_${TARGET}_${split_name}.log"

    printf "\n%b\n" "${BOLD}${BLUE}== ${TARGET} ${split_name} ==${RESET}"
    info "Detailed output: ${split_log}"

    run_logged "build single-feature datasets for ${TARGET} ${split_name}" "${split_log}" \
      "${PYTHON}" "${PROJECT_ROOT}/DataSet/scripts/build_ml_dataset.py" \
      --source-csv "${SOURCE_CSV}" \
      --target "${TARGET}" \
      --seed "${seed}" \
      --test-size "${test_size}" \
      --train-batches ${TRAIN_BATCHES} \
      --extra-batches ${EXTRA_BATCHES}

    run_logged "combine feature datasets for ${TARGET} ${split_name}" "${split_log}" \
      "${PYTHON}" "${PROJECT_ROOT}/DataSet/scripts/combine_ml_dataset.py" \
      --target "${TARGET}" \
      --seed "${seed}" \
      --test-size "${test_size}"

    run_logged "validate ML datasets for ${TARGET} ${split_name}" "${split_log}" \
      "${PYTHON}" "${PROJECT_ROOT}/DataSet/scripts/validate_ml_dataset.py" \
      --target "${TARGET}" \
      --seed "${seed}" \
      --test-size "${test_size}" \
      --source-csv "${SOURCE_CSV}"

    ok "Completed ${TARGET} ${split_name}"
  done
done

ok "All ML dataset splits completed."
info "Logs are in ${LOG_DIR}/${RUN_ID}_*.log"
