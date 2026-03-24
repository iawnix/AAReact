echo "Hyper: model[rf] descripe[rdkit_3] seed[1] test_size[0.2]"
python /home/iaw/DATA2/AAReact/src/AHO_train.py --task "train" \
    --model_config "/home/iaw/DATA2/AAReact/config/train_ml_rf_rdkit_3.toml"

echo "Hyper: model[rf] descripe[rdkit_soap_3] seed[1] test_size[0.2]"
python /home/iaw/DATA2/AAReact/src/AHO_train.py --task "train" \
    --model_config "/home/iaw/DATA2/AAReact/config/train_ml_rf_rdkit_soap_3.toml"

echo "Hyper: model[rf] descripe[rdkit_soap_xtb_3] seed[1] test_size[0.2]"
python /home/iaw/DATA2/AAReact/src/AHO_train.py --task "train" \
    --model_config "/home/iaw/DATA2/AAReact/config/train_ml_rf_rdkit_soap_xtb_3.toml"

echo "Hyper: model[rf] descripe[soap_xtb_3] seed[1] test_size[0.2]"
python /home/iaw/DATA2/AAReact/src/AHO_train.py --task "train" \
    --model_config "/home/iaw/DATA2/AAReact/config/train_ml_rf_soap_xtb_3.toml"


echo "Hyper: model[xgb] descripe[rdkit_3] seed[1] test_size[0.2]"
python /home/iaw/DATA2/AAReact/src/AHO_train.py --task "train" \
    --model_config "/home/iaw/DATA2/AAReact/config/train_ml_xgb_rdkit_3.toml"

echo "Hyper: model[xgb] descripe[rdkit_soap_3] seed[1] test_size[0.2]"
python /home/iaw/DATA2/AAReact/src/AHO_train.py --task "train" \
    --model_config "/home/iaw/DATA2/AAReact/config/train_ml_xgb_rdkit_soap_3.toml"

echo "Hyper: model[xgb] descripe[rdkit_soap_xtb_3] seed[1] test_size[0.2]"
python /home/iaw/DATA2/AAReact/src/AHO_train.py --task "train" \
    --model_config "/home/iaw/DATA2/AAReact/config/train_ml_xgb_rdkit_soap_xtb_3.toml"

echo "Hyper: model[xgb] descripe[soap_xtb_3] seed[1] test_size[0.2]"
python /home/iaw/DATA2/AAReact/src/AHO_train.py --task "train" \
    --model_config "/home/iaw/DATA2/AAReact/config/train_ml_xgb_soap_xtb_3.toml"

