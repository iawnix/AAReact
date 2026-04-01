echo "Train: model[rf] descripe[rdkit_3] seed[1] test_size[0.2]"
python /home/iaw/DATA2/AAReact/src/AHO_train.py --task "train" \
    --model_config "/home/iaw/DATA2/AAReact/config/train_ml_rf_rdkit_3.toml"

echo "Train: model[rf] descripe[acsf_3] seed[1] test_size[0.2] cv[5]"
python /home/iaw/DATA2/AAReact/src/AHO_train.py --task "train" \
        --model_config "/home/iaw/DATA2/AAReact/config/train_ml_rf_acsf_3.toml"
echo "#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#"
echo ""

echo "Train: model[rf] descripe[soap_3] seed[1] test_size[0.2] cv[5]"
python /home/iaw/DATA2/AAReact/src/AHO_train.py --task "train" \
        --model_config "/home/iaw/DATA2/AAReact/config/train_ml_rf_soap_3.toml"
echo "#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#"
echo ""

echo "Train: model[rf] descripe[xtb_3] seed[1] test_size[0.2] cv[5]"
python /home/iaw/DATA2/AAReact/src/AHO_train.py --task "train" \
        --model_config "/home/iaw/DATA2/AAReact/config/train_ml_rf_xtb_3.toml"
echo "#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#"
echo ""

echo "Train: model[rf] descripe[rdkit_soap_3] seed[1] test_size[0.2]"
python /home/iaw/DATA2/AAReact/src/AHO_train.py --task "train" \
    --model_config "/home/iaw/DATA2/AAReact/config/train_ml_rf_rdkit_soap_3.toml"

echo "Train: model[rf] descripe[rdkit_soap_xtb_3] seed[1] test_size[0.2]"
python /home/iaw/DATA2/AAReact/src/AHO_train.py --task "train" \
    --model_config "/home/iaw/DATA2/AAReact/config/train_ml_rf_rdkit_soap_xtb_3.toml"

echo "Train: model[rf] descripe[soap_xtb_3] seed[1] test_size[0.2]"
python /home/iaw/DATA2/AAReact/src/AHO_train.py --task "train" \
    --model_config "/home/iaw/DATA2/AAReact/config/train_ml_rf_soap_xtb_3.toml"


echo "Train: model[xgb] descripe[rdkit_3] seed[1] test_size[0.2]"
python /home/iaw/DATA2/AAReact/src/AHO_train.py --task "train" \
    --model_config "/home/iaw/DATA2/AAReact/config/train_ml_xgb_rdkit_3.toml"

echo "Train: model[xgb] descripe[acsf_3] seed[1] test_size[0.2] cv[5]"
python /home/iaw/DATA2/AAReact/src/AHO_train.py --task "train" \
        --model_config "/home/iaw/DATA2/AAReact/config/train_ml_xgb_acsf_3.toml"
echo "#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#"
echo ""

echo "Train: model[xgb] descripe[soap_3] seed[1] test_size[0.2] cv[5]"
python /home/iaw/DATA2/AAReact/src/AHO_train.py --task "train" \
        --model_config "/home/iaw/DATA2/AAReact/config/train_ml_xgb_soap_3.toml"
echo "#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#"
echo ""

echo "Train: model[xgb] descripe[xtb_3] seed[1] test_size[0.2] cv[5]"
python /home/iaw/DATA2/AAReact/src/AHO_train.py --task "train" \
        --model_config "/home/iaw/DATA2/AAReact/config/train_ml_xgb_xtb_3.toml"
echo "#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#"
echo ""

echo "Train: model[xgb] descripe[rdkit_soap_3] seed[1] test_size[0.2]"
python /home/iaw/DATA2/AAReact/src/AHO_train.py --task "train" \
    --model_config "/home/iaw/DATA2/AAReact/config/train_ml_xgb_rdkit_soap_3.toml"

echo "Train: model[xgb] descripe[rdkit_soap_xtb_3] seed[1] test_size[0.2]"
python /home/iaw/DATA2/AAReact/src/AHO_train.py --task "train" \
    --model_config "/home/iaw/DATA2/AAReact/config/train_ml_xgb_rdkit_soap_xtb_3.toml"

echo "Train: model[xgb] descripe[soap_xtb_3] seed[1] test_size[0.2]"
python /home/iaw/DATA2/AAReact/src/AHO_train.py --task "train" \
    --model_config "/home/iaw/DATA2/AAReact/config/train_ml_xgb_soap_xtb_3.toml"

echo "Train: model[lgb] descripe[rdkit_3] seed[1] test_size[0.2]"
python /home/iaw/DATA2/AAReact/src/AHO_train.py --task "train" \
    --model_config "/home/iaw/DATA2/AAReact/config/train_ml_lgb_rdkit_3.toml"

echo "Train: model[lgb] descripe[acsf_3] seed[1] test_size[0.2] cv[5]"
python /home/iaw/DATA2/AAReact/src/AHO_train.py --task "train" \
        --model_config "/home/iaw/DATA2/AAReact/config/train_ml_lgb_acsf_3.toml"
echo "#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#"
echo ""

echo "Train: model[lgb] descripe[soap_3] seed[1] test_size[0.2] cv[5]"
python /home/iaw/DATA2/AAReact/src/AHO_train.py --task "train" \
        --model_config "/home/iaw/DATA2/AAReact/config/train_ml_lgb_soap_3.toml"
echo "#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#"
echo ""

echo "Train: model[lgb] descripe[xtb_3] seed[1] test_size[0.2] cv[5]"
python /home/iaw/DATA2/AAReact/src/AHO_train.py --task "train" \
        --model_config "/home/iaw/DATA2/AAReact/config/train_ml_lgb_xtb_3.toml"
echo "#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#"
echo ""

echo "Train: model[lgb] descripe[rdkit_soap_3] seed[1] test_size[0.2]"
python /home/iaw/DATA2/AAReact/src/AHO_train.py --task "train" \
    --model_config "/home/iaw/DATA2/AAReact/config/train_ml_lgb_rdkit_soap_3.toml"

echo "Train: model[lgb] descripe[rdkit_soap_xtb_3] seed[1] test_size[0.2]"
python /home/iaw/DATA2/AAReact/src/AHO_train.py --task "train" \
    --model_config "/home/iaw/DATA2/AAReact/config/train_ml_lgb_rdkit_soap_xtb_3.toml"

echo "Train: model[lgb] descripe[soap_xtb_3] seed[1] test_size[0.2]"
python /home/iaw/DATA2/AAReact/src/AHO_train.py --task "train" \
    --model_config "/home/iaw/DATA2/AAReact/config/train_ml_lgb_soap_xtb_3.toml"
