#!/usr/bin/env bash
set -euo pipefail

TRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${TRAIN_DIR}/.." && pwd)"
PYTHON="${PYTHON:-/home/iaw/soft/conda/2024.06.1/envs/python3.12/bin/python}"

# Workflow controls. Edit these values here instead of passing CLI arguments.
TARGETS=("ddg")
HYPER_SPLIT_NAME="seed_1_test_0-2"
SEARCH_METHODS=("optuna")
SEEDS=(1 12 42)
TEST_SIZES=(0.15 0.2 0.3)
MODELS=("lgb" "xgb" "rf")
DESCRIPTORS=(
  "rdkit"
  "xtb"
  "soap"
  "acsf"
  "rdkit_soap"
  "soap_xtb"
  "rdkit_xtb"
  "soap_acsf"
  "acsf_xtb"
  "rdkit_acsf"
  "rdkit_soap_xtb"
  "rdkit_soap_acsf"
  "rdkit_xtb_acsf"
  "soap_xtb_acsf"
  "rdkit_soap_xtb_acsf"
)

fail() {
  printf '[FAIL] %s\n' "$*" >&2
}

info() {
  printf '[INFO] %s\n' "$*"
}

ok() {
  printf '[OK] %s\n' "$*"
}

setup_logging() {
  RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
  LOG_DIR="${LOG_DIR:-${TRAIN_DIR}/logs}"
  RUN_LOG="${RUN_LOG:-${LOG_DIR}/train_${RUN_ID}.log}"
  mkdir -p "$(dirname "${RUN_LOG}")"
  exec > >(tee -a "${RUN_LOG}") 2>&1
}

finish() {
  local status=$?
  if [ "${status}" -eq 0 ]; then
    ok "Train log saved: ${RUN_LOG}"
    ok "Extract metrics with: ${TRAIN_DIR}/2_extract_train_model_metric.sh ${RUN_LOG} ${TRAIN_DIR}/train_${RUN_ID}.csv"
  else
    fail "Training workflow failed with status ${status}. Log saved: ${RUN_LOG}"
  fi
}

split_name_for() {
  local seed="$1"
  local test_size="$2"
  printf 'seed_%s_test_%s' "${seed}" "${test_size//./-}"
}

setup_logging
trap finish EXIT

if [ "$#" -gt 0 ]; then
  fail "This script is controlled by variables inside the script, not by CLI arguments."
  fail "Edit TARGETS, HYPER_SPLIT_NAME, SEARCH_METHODS, SEEDS, TEST_SIZES, MODELS, or DESCRIPTORS in ${BASH_SOURCE[0]}."
  exit 2
fi

info "AAReact ML training workflow"
printf '  python: %s\n' "${PYTHON}"
printf '  targets: %s\n' "${TARGETS[*]}"
printf '  hyper_split: %s\n' "${HYPER_SPLIT_NAME}"
printf '  search_methods: %s\n' "${SEARCH_METHODS[*]}"
printf '  train_seeds: %s\n' "${SEEDS[*]}"
printf '  train_test_sizes: %s\n' "${TEST_SIZES[*]}"
printf '  models: %s\n' "${MODELS[*]}"
printf '  descriptors: %s\n' "${DESCRIPTORS[*]}"
printf '  config_root: %s\n' "${PROJECT_ROOT}/config/train"
printf '  log: %s\n' "${RUN_LOG}"

total_runs=0
for target in "${TARGETS[@]}"; do
  case "${target}" in
    ddg|ee)
      ;;
    *)
      fail "Unsupported target in TARGETS: ${target}"
      exit 2
      ;;
  esac

  for search_method in "${SEARCH_METHODS[@]}"; do
    case "${search_method}" in
      grid|optuna)
        ;;
      *)
        fail "Unsupported search method in SEARCH_METHODS: ${search_method}"
        exit 2
        ;;
    esac

    for seed in "${SEEDS[@]}"; do
      for test_size in "${TEST_SIZES[@]}"; do
        train_split_name="$(split_name_for "${seed}" "${test_size}")"
        config_dir="${PROJECT_ROOT}/config/train/${target}/hyper_${HYPER_SPLIT_NAME}/search_${search_method}/${train_split_name}"

        if [ ! -d "${config_dir}" ]; then
          fail "Train config split does not exist: ${config_dir}"
          fail "Generate configs first with config/2_generate_train_tomls.sh."
          exit 1
        fi

        for model in "${MODELS[@]}"; do
          for desc_type in "${DESCRIPTORS[@]}"; do
            config_fp="${config_dir}/train_ml_${model}_${desc_type}.toml"
            if [ ! -f "${config_fp}" ]; then
              fail "Missing train config: ${config_fp}"
              exit 1
            fi

            info "Train target=${target} hyper=${HYPER_SPLIT_NAME} search=${search_method} split=${train_split_name} model=${model} descriptor=${desc_type}"
            "${PYTHON}" "${PROJECT_ROOT}/src/AHO_train.py" \
              --task "train" \
              --model_config "${config_fp}" \
              --verbose True
            ok "Finished target=${target} search=${search_method} split=${train_split_name} model=${model} descriptor=${desc_type}"
            total_runs=$((total_runs + 1))
          done
        done
      done
    done
  done
done

ok "Finished ${total_runs} training runs."
