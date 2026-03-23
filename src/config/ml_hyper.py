from typing import Union, List, Any
from dataclasses import dataclass
import tomllib
from dacite import from_dict

@dataclass
class hyper_params:
    data_x: str
    data_y: str
    x_label: str
    data_class: str
    seed: int
    test_size: float                    # 对于机器学习模型, 没有划分valid
    cv: str

class ml_hyper_config:
    Model_type: str
    Hyper: hyper_params

def init_config_from_hyper_toml(toml_fp: str) -> ml_hyper_config:
    with open(toml_fp, "rb") as F:
        ss = tomllib.load(F)

    model_type = ss.get("Model").get("name")
    hyper_config = from_dict(hyper_params, ss.get("Hyper"))
    return ml_hyper_config(
        Model_type = model_type, 
        Hyper = hyper_config
    )
