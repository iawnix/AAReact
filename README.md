# IAWNIX
- 20260112
- iawhaha@163.com
- atropic acid 氢化还原反应的手性选择性预测
# 数据格式:
1. ee计算
- $ee = \frac{|R - S|}{R + S}$

# Install
- 需要通过xtb计算半经验的分子描述符, 因此, 你在安装前必须根据实际情况, 修改`src/util/constants.py`文件
    * `XTB_BACHEND`: 指定xtb的可执行路径
    * `OBABEL_BACHEND`: 指定openbabel的可执行路径
    * `XTB_WORK_SCRATCH`: 指定XTB计算产生临时文件的路径, 该路径在xtb计算过程中会不断的被覆盖写


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

   
