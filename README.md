# AAReact
- Date: 20260112
- Email: iawhaha@163.com
- Purpose: Prediction of Enantioselectivity in the Hydrogenation of Atropic Acid
- Author: Iawnix

# Data Format Requirements:
1. ee
- $ee = \frac{|R - S|}{R + S}$

# Install
- Semi-empirical molecular descriptors are computed via xTB. Modify `src/util/constants.py` before installation:
    * `XTB_BACHEND`: Path to the xTB executable
    * `OBABEL_BACHEND`: Path to the OpenBabel executable
    * `XTB_WORK_SCRATCH`: Scratch directory for xTB temporary files (continuously overwritten)
- `conda create -n AAReact python=3.12`
- `conda activate AAReact`
- `pip install .`

# Calc Feat
1. Generate initial features
```
python /home/iaw/DATA2/AAReact/src/AHO_calc_raw_features.py \
    --in_csv "/home/iaw/DATA2/AAReact/DataSet/Data_All/full_data_464.csv" \
    --desc_type "all" \
    --sdf_path "/home/iaw/DATA2/AAReact/DataSet/Data_All/1_sdf" \
    --save_path "/home/iaw/DATA2/AAReact/DataSet/Data_All/2_raw_features"
```

# Train
```
AHO_train.py --task "train" \
        --model_config "/home/iaw/DATA2/AAReact/config/train_ml_xgb.toml"
```
- more information, see `config/train_*.toml`
# Hyper
```
AHO_train.py --task "hyper" \
        --model_config "/home/iaw/DATA2/AAReact/config/train_ml_xgb.toml"
```
- More information, see `config/hyper_*.toml`
- The parameters for grid hyperparameters are defined in `src/util/constants.py`. Please reinstall the code if you need to modify any variables therein.


