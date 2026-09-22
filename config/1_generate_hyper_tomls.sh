#!/usr/bin/env bash
set -euo pipefail

CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${CONFIG_DIR}/.." && pwd)"
PYTHON="${PYTHON:-/home/iaw/soft/conda/2024.06.1/envs/python3.12/bin/python}"

# Workflow controls. Edit these values here instead of passing CLI arguments.
TARGETS=("ddg")
SEEDS=(1)
TEST_SIZES=(0.2)
SEARCH_METHODS=("grid" "optuna")
BATCH_TYPE="train_test"
CV=5
N_TRIALS=400
N_STARTUP_TRIALS=20
OBJECTIVE_STD_PENALTY=0.1
TRAIN_GAP_PENALTY=0.0

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
  fail "Edit TARGETS, SEEDS, TEST_SIZES, SEARCH_METHODS, or BATCH_TYPE in ${BASH_SOURCE[0]}."
  exit 2
fi

info "Generate AAReact hyper TOML files"
printf '  python: %s\n' "${PYTHON}"
printf '  targets: %s\n' "${TARGETS[*]}"
printf '  seeds: %s\n' "${SEEDS[*]}"
printf '  test_sizes: %s\n' "${TEST_SIZES[*]}"
printf '  search_methods: %s\n' "${SEARCH_METHODS[*]}"
printf '  batch_type: %s\n' "${BATCH_TYPE}"
printf '  cv: %s\n' "${CV}"
printf '  n_trials: %s\n' "${N_TRIALS}"
printf '  objective_std_penalty: %s\n' "${OBJECTIVE_STD_PENALTY}"
printf '  output_root: %s\n' "${CONFIG_DIR}/hyper"

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

  for seed in "${SEEDS[@]}"; do
    for test_size in "${TEST_SIZES[@]}"; do
      split_name="$(split_name_for "${seed}" "${test_size}")"
      dataset_dir="${PROJECT_ROOT}/DataSet/Data_All/3_data_for_train/${target}/${split_name}"

      if [ ! -d "${dataset_dir}" ]; then
        fail "Dataset split does not exist: ${dataset_dir}"
        fail "Build it first with DataSet/Data_All/2_build_ml_datasets.sh."
        exit 1
      fi

      for search_method in "${SEARCH_METHODS[@]}"; do
        case "${search_method}" in
          grid|optuna)
            ;;
          *)
            fail "Unsupported search method in SEARCH_METHODS: ${search_method}"
            exit 2
            ;;
        esac

        info "Generate ${target} ${split_name} search=${search_method}"
        "${PYTHON}" "${CONFIG_DIR}/script/gen_hyper_toml.py" \
          --target "${target}" \
          --seed "${seed}" \
          --test-size "${test_size}" \
          --split-name "${split_name}" \
          --batch-type "${BATCH_TYPE}" \
          --search-method "${search_method}" \
          --cv "${CV}" \
          --n-trials "${N_TRIALS}" \
          --n-startup-trials "${N_STARTUP_TRIALS}" \
          --objective-std-penalty "${OBJECTIVE_STD_PENALTY}" \
          --train-gap-penalty "${TRAIN_GAP_PENALTY}"
        generated_count="$(find "${CONFIG_DIR}/hyper/${target}/${split_name}/${search_method}" -maxdepth 1 -type f -name 'hyper_ml_*.toml' | wc -l | tr -d ' ')"
        ok "Generated ${generated_count} files for ${target} ${split_name} search=${search_method}"
      done
      total_splits=$((total_splits + 1))
    done
  done
done

ok "Generated hyper TOML files for ${total_splits} dataset splits under ${CONFIG_DIR}/hyper"
