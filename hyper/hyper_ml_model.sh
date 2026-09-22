#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-/home/iaw/soft/conda/2024.06.1/envs/python3.12/bin/python}"
AHO_TRAIN="${PROJECT_ROOT}/src/AHO_train.py"
CONFIG_ROOT="${PROJECT_ROOT}/config/hyper"
HYPER_LOG_DIR="${PROJECT_ROOT}/hyper/output/hyper_log"

# Workflow controls. Edit these values here instead of passing CLI arguments.
TARGETS=("ddg")
SEEDS=(1)
TEST_SIZES=(0.2)
SEARCH_METHODS=("optuna")
MODELS=("lgb" "xgb" "rf")
DESCRIPTORS=(
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
  "rdkit"
  "xtb"
  "acsf"
  "soap"
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

split_name_for() {
  local seed="$1"
  local test_size="$2"
  printf 'seed_%s_test_%s' "${seed}" "${test_size//./-}"
}

if [ "$#" -gt 0 ]; then
  fail "This script is controlled by variables inside the script, not by CLI arguments."
  fail "Edit TARGETS, SEEDS, TEST_SIZES, SEARCH_METHODS, MODELS, or DESCRIPTORS in ${BASH_SOURCE[0]}."
  exit 2
fi

mkdir -p "${HYPER_LOG_DIR}"

info "AAReact ML hyperparameter workflow"
printf '  python: %s\n' "${PYTHON}"
printf '  targets: %s\n' "${TARGETS[*]}"
printf '  seeds: %s\n' "${SEEDS[*]}"
printf '  test_sizes: %s\n' "${TEST_SIZES[*]}"
printf '  search_methods: %s\n' "${SEARCH_METHODS[*]}"
printf '  models: %s\n' "${MODELS[*]}"
printf '  descriptors: %s\n' "${DESCRIPTORS[*]}"
printf '  config_root: %s\n' "${CONFIG_ROOT}"
printf '  hyper_log_dir: %s\n' "${HYPER_LOG_DIR}"

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

  for seed in "${SEEDS[@]}"; do
    for test_size in "${TEST_SIZES[@]}"; do
      split_name="$(split_name_for "${seed}" "${test_size}")"
      for search_method in "${SEARCH_METHODS[@]}"; do
        case "${search_method}" in
          grid|optuna)
            ;;
          *)
            fail "Unsupported search method in SEARCH_METHODS: ${search_method}"
            exit 2
            ;;
        esac

        mkdir -p "${HYPER_LOG_DIR}/${search_method}"
        config_dir="${CONFIG_ROOT}/${target}/${split_name}/${search_method}"
        if [ ! -d "${config_dir}" ]; then
          fail "Missing hyper config directory: ${config_dir}"
          fail "Generate it first with config/1_generate_hyper_tomls.sh."
          exit 1
        fi

        for model in "${MODELS[@]}"; do
          for descriptor in "${DESCRIPTORS[@]}"; do
            model_config="${config_dir}/hyper_ml_${model}_${descriptor}.toml"
            if [ ! -s "${model_config}" ]; then
              fail "Missing hyper config: ${model_config}"
              exit 1
            fi

            printf '========================================================================\n'
            printf 'Hyper: target[%s] split[%s] search[%s] model[%s] descriptor[%s]\n' \
              "${target}" "${split_name}" "${search_method}" "${model}" "${descriptor}"
            printf '========================================================================\n'

            "${PYTHON}" "${AHO_TRAIN}" \
              --task "hyper" \
              --model_config "${model_config}"

            printf '#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#\n\n'
            total_runs=$((total_runs + 1))
          done
        done
      done

      ok "Completed ${target} ${split_name}"
    done
  done
done

ok "Completed ${total_runs} hyperparameter runs."
