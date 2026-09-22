from typing import Union, List
from dataclasses import dataclass
import tomllib
from dacite import from_dict

from config.constants import normalize_target

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
    gamma: float = 0.0

@dataclass
class lgb_params:
    colsample_bytree: float
    learning_rate: float
    max_depth: int
    min_child_samples: int
    n_estimators: int
    num_leaves: int
    reg_alpha: float
    reg_lambda: float
    subsample: float
    min_split_gain: float = 0.0

@dataclass
class rf_params:
    max_depth: Union[int, None]
    min_samples_leaf: int
    min_samples_split: int
    n_estimators: int
    ccp_alpha: float
    max_features: Union[float, str, None] = 1.0
    bootstrap: bool = True

@dataclass
class train_params:
    data_x: str
    data_y: str
    x_label: str
    data_class: str
    seed: int
    test_size: float                    # 对于机器学习模型, 没有划分valid
    model_save: str
    n_cpu: int                          # 新增模型训练CPU核数
    target: str = "ee"


@dataclass
class ml_trian_config:
    Model_type: str
    Model: Union[xgb_params, rf_params, lgb_params]
    Train: train_params

MODEL_CONFIG_MAP = {
    "rf": rf_params, 
    "xgb": xgb_params,
    "lgb": lgb_params
}

def normalize_model_params(model_parms: dict) -> dict:
    out = dict(model_parms)
    for key, value in list(out.items()):
        if not isinstance(value, str):
            continue
        text = value.strip()
        lower = text.lower()
        if lower in ("none", "null"):
            out[key] = None
        elif lower == "true":
            out[key] = True
        elif lower == "false":
            out[key] = False
    return out

def init_config_from_train_toml(toml_fp: str) -> ml_trian_config:
    with open(toml_fp, "rb") as F:
        ss = tomllib.load(F)
    
    model_type = ss.get("Model").get("model_type")
    model_parms = {k: v for k, v in ss.get("Model").items() if k != "model_type"}
    model_parms = normalize_model_params(model_parms)
    
    model_config = from_dict(MODEL_CONFIG_MAP[model_type], model_parms)
    train_dict = dict(ss.get("Train"))
    train_dict["target"] = normalize_target(train_dict.get("target", "ee"))
    train_config = from_dict(train_params, train_dict)
    return ml_trian_config(
            Model_type = model_type, 
            Model = model_config, 
            Train = train_config
        )
