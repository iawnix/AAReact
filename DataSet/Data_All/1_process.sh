#!/usr/bin/env bash
set -euo pipefail

PYTHON="/home/iaw/soft/conda/2024.06.1/envs/python3.12/bin/python"
SAVE_PATH="/home/iaw/DATA2/AAReact/DataSet/Data_All/2_raw_features"

mkdir -p "${SAVE_PATH}"

"${PYTHON}" /home/iaw/DATA2/AAReact/src/AHO_calc_raw_features.py \
    --in_csv "/home/iaw/DATA2/AAReact/DataSet/Data_All/full_data_436-20260624.csv" \
    --desc_type "all" \
    --sdf_path "/home/iaw/DATA2/AAReact/DataSet/Data_All/1_sdf" \
    --save_path "${SAVE_PATH}"
