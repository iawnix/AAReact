from typing import Union, List
from dataclasses import dataclass
import tomllib
from dacite import from_dict

@dataclass
class xgb_params:
    colsample_bytree: float
    learning_rate: float
    max_depth: int
    min_child_weight: int
    n_estimators: int
    reg_alpha: float
    reg_lambda: float
    subsample: float

@dataclass
class rf_params:
    max_depth: int
    min_samples_leaf: int
    min_samples_split: int
    n_estimators: int
    ccp_alpha: float

@dataclass
class train_params:
    data_x: str
    data_y: str
    x_label: str
    data_class: str
    seed: int
    test_size: float                    # 对于机器学习模型, 没有划分valid
    model_save: str


@dataclass
class ml_trian_config:
    Model_type: str
    Model: Union[xgb_params, rf_params]
    Train: train_params

MODEL_CONFIG_MAP = {
    "rf": rf_params, 
    "xgb": xgb_params
}

def init_config_from_train_toml(toml_fp: str) -> ml_trian_config:
    with open(toml_fp, "rb") as F:
        ss = tomllib.load(F)
    
    model_type = ss.get("Model").get("model_type")
    model_parms = {k: v for k, v in ss.get("Model").items() if k != "model_type"}
    
    model_config = from_dict(MODEL_CONFIG_MAP[model_type], model_parms)
    train_config = from_dict(train_params, ss.get("Train"))
    return ml_trian_config(
            Model_type = model_type, 
            Model = model_config, 
            Train = train_config
        )
