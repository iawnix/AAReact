import yaml
import toml
from typing import Dict, Tuple
from types import SimpleNamespace

import os
import torch
import numpy as np
import pandas as pd
import random
import pytorch_lightning as pl



"""
这是一个旧脚本, 用来训练dl模型的
"""

def set_seed(seed: int) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    pl.seed_everything(seed, workers=True, verbose=False)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def double_dict_to_namespace(d: Dict) -> SimpleNamespace:
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = double_dict_to_namespace(value)
    return SimpleNamespace(**d)


def namespace_to_double_dict(ns: SimpleNamespace) -> dict:

    result = vars(ns).copy()
    
    for key, value in result.items():
        if isinstance(value, SimpleNamespace):
            result[key] = namespace_to_double_dict(value)
        elif isinstance(value, list):
            result[key] = [
                namespace_to_double_dict(item) if isinstance(item, SimpleNamespace) else item
                for item in value
            ]
    
    return result

def init_config_yaml(config_fp: str) -> Dict:

    with open(config_fp, 'r') as f:
        config = yaml.safe_load(f)

    return double_dict_to_namespace(config)

def init_config_toml(config_fp: str):
    with open(config_fp, 'r') as f:
        config = toml.load(f)
    
    return double_dict_to_namespace(config)