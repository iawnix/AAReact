#!/usr/bin/bash

model_s=("lgb" "xgb" "rf")
comb_s=(
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
# 双层循环
for i_m in "${model_s[@]}"; do
    for i_desc_type in "${comb_s[@]}"; do
        echo "========================================================================"
        echo "Hyper: model[$i_m] descripe[$i_desc_type] seed[1] test_size[0.2] cv[5]"
        echo "========================================================================"

        # 运行 Python 脚本（纯 bash 写法）
        python /home/iaw/DATA/AAReact/src/AHO_train.py \
            --task "hyper" \
            --model_config "/home/iaw/DATA/AAReact/config/hyper_ml_${i_m}_${i_desc_type}.toml"
        echo "#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#"
        echo ""
    done
done
