
echo "Hyper: model[rf] descripe[rdkit_3] seed[1] test_size[0.2] cv[5]"
python /home/iaw/DATA2/AAReact/src/AHO_train.py --task "hyper" \
        --model_config "/home/iaw/DATA2/AAReact/config/hyper_ml_rf_rdkit_3.toml"
echo "#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#"
echo ""


echo "Hyper: model[rf] descripe[rdkit_soap_3] seed[1] test_size[0.2] cv[5]"
python /home/iaw/DATA2/AAReact/src/AHO_train.py --task "hyper" \
        --model_config "/home/iaw/DATA2/AAReact/config/hyper_ml_rf_rdkit_soap_3.toml"
echo "#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#"
echo ""

echo "Hyper: model[rf] descripe[rdkit_soap_xtb_3] seed[1] test_size[0.2] cv[5]"
python /home/iaw/DATA2/AAReact/src/AHO_train.py --task "hyper" \
        --model_config "/home/iaw/DATA2/AAReact/config/hyper_ml_rf_rdkit_soap_xtb_3.toml"
echo "#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#"
echo ""
