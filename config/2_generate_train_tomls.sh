#!/usr/bin/env bash
set -euo pipefail

CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${CONFIG_DIR}/.." && pwd)"
PYTHON="${PYTHON:-/home/iaw/soft/conda/2024.06.1/envs/python3.12/bin/python}"

# Workflow controls. Edit these values here instead of passing CLI arguments.
TARGETS=("ddg")
HYPER_SPLIT_NAME="seed_1_test_0-2"
SEARCH_METHODS=("optuna")
SEEDS=(1 12 42)
TEST_SIZES=(0.15 0.2 0.3)
BATCH_TYPE="train_test"
CV=5
N_CPU=5

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
  fail "Edit TARGETS, HYPER_SPLIT_NAME, SEARCH_METHODS, SEEDS, TEST_SIZES, BATCH_TYPE, CV, or N_CPU in ${BASH_SOURCE[0]}."
  exit 2
fi

info "Generate AAReact train TOML files"
printf '  python: %s\n' "${PYTHON}"
printf '  targets: %s\n' "${TARGETS[*]}"
printf '  hyper_split: %s\n' "${HYPER_SPLIT_NAME}"
printf '  search_methods: %s\n' "${SEARCH_METHODS[*]}"
printf '  train_seeds: %s\n' "${SEEDS[*]}"
printf '  train_test_sizes: %s\n' "${TEST_SIZES[*]}"
printf '  batch_type: %s\n' "${BATCH_TYPE}"
printf '  cv: %s\n' "${CV}"
printf '  n_cpu: %s\n' "${N_CPU}"
printf '  output_root: %s\n' "${CONFIG_DIR}/train"

total_splits=0
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

    hyper_log_dir="${PROJECT_ROOT}/hyper/output/hyper_log/${search_method}"
    if [ -d "${hyper_log_dir}" ]; then
      hyper_log_count="$(find "${hyper_log_dir}" -maxdepth 1 -type f -name "*_${target}_${HYPER_SPLIT_NAME}_cv_${CV}_hyper.log" | wc -l | tr -d ' ')"
    else
      hyper_log_count=0
    fi
    if [ "${hyper_log_count}" -eq 0 ]; then
      fail "No hyper logs found for target=${target}, hyper_split=${HYPER_SPLIT_NAME}, search=${search_method}, cv=${CV}."
      fail "Run hyper/hyper_ml_model.sh first."
      exit 1
    fi

    for seed in "${SEEDS[@]}"; do
      for test_size in "${TEST_SIZES[@]}"; do
        train_split_name="$(split_name_for "${seed}" "${test_size}")"
        dataset_dir="${PROJECT_ROOT}/DataSet/Data_All/3_data_for_train/${target}/${train_split_name}"

        if [ ! -d "${dataset_dir}" ]; then
          fail "Dataset split does not exist: ${dataset_dir}"
          fail "Build it first with DataSet/Data_All/2_build_ml_datasets.sh."
          exit 1
        fi

        info "Generate ${target} hyper=${HYPER_SPLIT_NAME} search=${search_method} train=${train_split_name}"
        "${PYTHON}" "${CONFIG_DIR}/script/gen_train_toml.py" \
          --target "${target}" \
          --hyper-split-name "${HYPER_SPLIT_NAME}" \
          --search-method "${search_method}" \
          --train-split-name "${train_split_name}" \
          --seed "${seed}" \
          --test-size "${test_size}" \
          --batch-type "${BATCH_TYPE}" \
          --cv "${CV}" \
          --n-cpu "${N_CPU}"

        generated_count="$(find "${CONFIG_DIR}/train/${target}/hyper_${HYPER_SPLIT_NAME}/search_${search_method}/${train_split_name}" -maxdepth 1 -type f -name 'train_ml_*.toml' | wc -l | tr -d ' ')"
        ok "Generated ${generated_count} train TOML files for ${target} search=${search_method} ${train_split_name}"
        total_splits=$((total_splits + 1))
      done
    done
  done
done

ok "Generated train TOML files for ${total_splits} dataset splits under ${CONFIG_DIR}/train"
