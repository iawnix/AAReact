from typing import Union, List, Any
from dataclasses import dataclass
import tomllib
from dacite import from_dict

from config.constants import normalize_target

SUPPORTED_SEARCH_METHODS = {"grid", "optuna"}

@dataclass
class hyper_params:
    data_x: str
    data_y: str
    x_label: str
    data_class: str
    seed: int
    test_size: float                    # 对于机器学习模型, 没有划分valid
    cv: int
    n_cpu: int                          # 新增优化器cpu核数
    target: str = "ee"

@dataclass
class search_params:
    method: str = "grid"
    metric: str = "rmse"
    cv: int = 5
    shuffle_cv: bool = True
    n_trials: int = 150
    n_startup_trials: int = 20
    objective_std_penalty: float = 0.1
    train_gap_penalty: float = 0.0
    study_dir: str = ""

@dataclass
class ml_hyper_config:
    Model_type: str
    n_cpu: int                          # 新增优化器cpu核数
    Hyper: hyper_params
    Search: search_params
    params_save: str

def init_config_from_hyper_toml(toml_fp: str) -> ml_hyper_config:
    with open(toml_fp, "rb") as F:
        ss = tomllib.load(F)

    model_type = ss.get("Model").get("name")
    save_fp = ss.get("Model").get("params_save")
    model_n_cpu = ss.get("Model").get("n_cpu")
    hyper_dict = dict(ss.get("Hyper"))
    hyper_dict["target"] = normalize_target(hyper_dict.get("target", "ee"))
    hyper_config = from_dict(hyper_params, hyper_dict)
    search_dict = dict(ss.get("Search") or {})
    search_dict.setdefault("cv", hyper_config.cv)
    search_dict["method"] = str(search_dict.get("method", "grid")).lower()
    search_dict["metric"] = str(search_dict.get("metric", "rmse")).lower()
    if search_dict["method"] not in SUPPORTED_SEARCH_METHODS:
        raise ValueError("Unsupported search method: {}. Supported methods: {}".format(
            search_dict["method"], ", ".join(sorted(SUPPORTED_SEARCH_METHODS))
        ))
    search_config = from_dict(search_params, search_dict)
    return ml_hyper_config(
        Model_type = model_type, 
        n_cpu = model_n_cpu, 
        Hyper = hyper_config, 
        Search = search_config,
        params_save = save_fp
    )
