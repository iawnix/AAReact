#!/usr/bin/env bash
# 使用 Data_All 的 DDG 模型和同一划分的特征标签。原反应条件保持不变。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-/home/iaw/soft/conda/2024.06.1/envs/python3.12/bin/python}"
SPLIT="${SPLIT:-seed_1_test_0-2}"
HYPER_SPLIT="${HYPER_SPLIT:-seed_1_test_0-2}"
SEARCH_METHOD="${SEARCH_METHOD:-optuna}"
SDF_DIR="${SDF_DIR:-${PROJECT_ROOT}/DataSet/Data_All/1_sdf}"
MODEL_DIR="${PROJECT_ROOT}/train/output/Data_All/pt/ddg/hyper_${HYPER_SPLIT}/search_${SEARCH_METHOD}/${SPLIT}"
FEATURE_DIR="${PROJECT_ROOT}/DataSet/Data_All/3_data_for_train/ddg/${SPLIT}"
read -r -a model_names <<< "${MODELS:-rf_rdkit rf_rdkit_soap rf_rdkit_soap_xtb rf_soap_xtb xgb_rdkit xgb_rdkit_soap xgb_rdkit_soap_xtb xgb_soap_xtb}"

if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--check-only" ) ]]; then
  printf 'Usage: bash %s [--check-only]\n' "${BASH_SOURCE[0]}" >&2
  exit 2
fi
# 核对模型目标、特征数量及顺序；先检查全部配置，避免中途才发现缺失。
"${PYTHON}" - "${MODEL_DIR}" "${FEATURE_DIR}" "${model_names[@]}" <<'CHECK'
import pickle
import sys
from pathlib import Path
from joblib import load

model_dir, feature_dir = map(Path, sys.argv[1:3])
for name in sys.argv[3:]:
    algorithm, descriptor = name.split('_', 1)
    if algorithm not in {'rf', 'xgb'}:
        raise ValueError(f'Unsupported legacy model: {name}')
    model = load(model_dir / f'{name}.pkl')
    with (feature_dir / descriptor / 'train_test_x_label.pkl').open('rb') as f:
        labels = pickle.load(f)
    if getattr(model, 'aa_target', None) != 'ddg':
        raise ValueError(f'{name}: expected a DDG model')
    if getattr(model, 'aa_x_label', None) != labels:
        raise ValueError(f'{name}: model and dataset feature labels differ')
    if model.n_features_in_ != sum(map(len, labels.values())):
        raise ValueError(f'{name}: feature count mismatch')
print(f'[OK] {len(sys.argv[3:])} models and their DDG feature labels match')
CHECK

# 分子结构核对：旧 CAT-71/72/73 -> 当前 CAT-22/39/38。
# 旧脚本 CAT-72/73 使用 SOL-2，仍沿用这一条件（不同于现有 extra 的 SOL-1）。
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/iter_1/new_cat_ddg/${SPLIT}}"
reactants=(REA-1 REA-2 REA-3 REA-4)
combinations=("CAT-22 SOL-1 20" "CAT-22 SOL-1 100" "CAT-39 SOL-2 20" "CAT-38 SOL-2 20")

# 在调用分子计算前检查所有结构路径。
for rea in "${reactants[@]}"; do
  [[ -s "${SDF_DIR}/${rea}.sdf" ]] || { printf '[FAIL] Missing %s\n' "${SDF_DIR}/${rea}.sdf" >&2; exit 1; }
done
for combination in "${combinations[@]}"; do
  read -r cat sol pressure <<< "${combination}"
  for molecule in "${cat}" "${sol}"; do
    [[ -s "${SDF_DIR}/${molecule}.sdf" ]] || { printf '[FAIL] Missing %s\n' "${SDF_DIR}/${molecule}.sdf" >&2; exit 1; }
  done
done
printf '[OK] Structures checked; target=ddg; split=%s; output=%s\n' "${SPLIT}" "${OUTPUT_DIR}"
if [[ "${1:-}" == "--check-only" ]]; then exit 0; fi

mkdir -p "${OUTPUT_DIR}"
# xTB 共用临时目录：本脚本顺序执行，不要与其他分子计算并发运行。
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
for model_name in "${model_names[@]}"; do
  descriptor="${model_name#*_}"
  for combination in "${combinations[@]}"; do
    read -r cat sol pressure <<< "${combination}"
    for rea in "${reactants[@]}"; do
      sample="${model_name}-${rea}_${sol}_${cat}_T60_P${pressure}"
      printf '%s\n' "${sample}"
      "${PYTHON}" "${PROJECT_ROOT}/src/AHO_predict.py" \
        --task ddg \
        --rea "${SDF_DIR}/${rea}.sdf" \
        --sol "${SDF_DIR}/${sol}.sdf" \
        --cat "${SDF_DIR}/${cat}.sdf" \
        --temp 60 --pressure "${pressure}" \
        --model "${MODEL_DIR}/${model_name}.pkl" \
        --feat_label "${FEATURE_DIR}/${descriptor}/train_test_x_label.pkl" \
        --verbose 0 --save_feat "${OUTPUT_DIR}/${sample}.npy" \
        | tee "${OUTPUT_DIR}/${sample}.log"
    done
  done
done
