from typing import Any, Dict
from unimol_tools import UniMolRepr
import numpy as np
from numpy.typing import NDArray

def condition_featurizer():
    pass

def mol_featurizer(model_size: str) -> NDArray:
    clf = UniMolRepr(data_type='molecule', 
        remove_hs=False,
        model_name="unimolv2",
        model_size=model_size,
        use_cuda=True
        )
    return clf

    