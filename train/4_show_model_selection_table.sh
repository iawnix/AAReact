#!/usr/bin/env bash
set -euo pipefail

TRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${TRAIN_DIR}/.." && pwd)"
PYTHON="${PYTHON:-/home/iaw/soft/conda/2024.06.1/envs/python3.12/bin/python}"

INPUT_CSV="${INPUT_CSV:-${TRAIN_DIR}/train.csv}"
OUTPUT_CSV="${OUTPUT_CSV:-${TRAIN_DIR}/Analysis/train_metrics/model_metrics_table.csv}"
VIEW="${VIEW:-primary}"
PRIMARY_SEED="${PRIMARY_SEED:-1}"
PRIMARY_TEST_SIZE="${PRIMARY_TEST_SIZE:-0.2}"
SORT_BLOCKS="${SORT_BLOCKS:-rmse_mean}"
SORT_ROWS="${SORT_ROWS:-seed_test}"
TOP_BLOCKS="${TOP_BLOCKS:-${TOP:-0}}"
TOP_ROWS="${TOP_ROWS:-0}"
COLOR="${COLOR:-auto}"
FULL="${FULL:-0}"
SAVE="${SAVE:-0}"
USE_PAGER="${USE_PAGER:-auto}"
PAGER_CMD="${PAGER:-less}"
TMP_REPORT=""

cleanup() {
  if [[ -n "${TMP_REPORT}" && -f "${TMP_REPORT}" ]]; then
    rm -f "${TMP_REPORT}"
  fi
}
trap cleanup EXIT

pager_command_available() {
  local first_word
  first_word="${PAGER_CMD%% *}"
  command -v "${first_word}" >/dev/null 2>&1
}

should_use_pager() {
  case "${USE_PAGER}" in
    0|false|False|FALSE|never|Never|NEVER|no|No|NO)
      return 1
      ;;
    1|true|True|TRUE|always|Always|ALWAYS|yes|Yes|YES)
      ;;
    auto|Auto|AUTO)
      [[ -t 1 ]] || return 1
      ;;
    *)
      printf '[FAIL] Invalid USE_PAGER: %s\n' "${USE_PAGER}" >&2
      exit 1
      ;;
  esac

  [[ "${PAGER_CMD}" != "cat" ]] || return 1
  pager_command_available || return 1
  return 0
}

REPORT_COLOR="${COLOR}"
USE_REPORT_PAGER=0
if should_use_pager; then
  USE_REPORT_PAGER=1
  if [[ "${REPORT_COLOR}" == "auto" ]]; then
    REPORT_COLOR="always"
  fi
fi

ARGS=(
  --input "${INPUT_CSV}"
  --output-csv "${OUTPUT_CSV}"
  --view "${VIEW}"
  --primary-seed "${PRIMARY_SEED}"
  --primary-test-size "${PRIMARY_TEST_SIZE}"
  --sort-blocks "${SORT_BLOCKS}"
  --sort-rows "${SORT_ROWS}"
  --top-blocks "${TOP_BLOCKS}"
  --top "${TOP_ROWS}"
  --color "${REPORT_COLOR}"
)

if [[ "${FULL}" == "1" || "${FULL}" == "true" ]]; then
  ARGS+=(--full)
fi
if [[ "${SAVE}" == "1" || "${SAVE}" == "true" ]]; then
  ARGS+=(--save)
fi
if [[ -n "${MODEL:-}" ]]; then
  ARGS+=(--model-name "${MODEL}")
fi
if [[ -n "${MODEL_NAME:-}" ]]; then
  ARGS+=(--model-name "${MODEL_NAME}")
fi
if [[ -n "${DESCRIPT:-}" ]]; then
  ARGS+=(--descriptor "${DESCRIPT}")
fi
if [[ -n "${DESCRIPTOR:-}" ]]; then
  ARGS+=(--descriptor "${DESCRIPTOR}")
fi
if [[ -n "${SEED:-}" ]]; then
  ARGS+=(--seed "${SEED}")
fi
if [[ -n "${TEST_SIZE:-}" ]]; then
  ARGS+=(--test-size "${TEST_SIZE}")
fi

cd "${PROJECT_ROOT}"

if [[ "${USE_REPORT_PAGER}" == "1" ]]; then
  TMP_REPORT="$(mktemp)"
  "${PYTHON}" "${TRAIN_DIR}/script/show_model_selection_table.py" "${ARGS[@]}" "$@" > "${TMP_REPORT}"
  if [[ "${PAGER_CMD}" == "less" ]]; then
    less -R -S "${TMP_REPORT}"
  else
    read -r -a PAGER_ARGS <<< "${PAGER_CMD}"
    "${PAGER_ARGS[@]}" "${TMP_REPORT}"
  fi
else
  exec "${PYTHON}" "${TRAIN_DIR}/script/show_model_selection_table.py" "${ARGS[@]}" "$@"
fi
