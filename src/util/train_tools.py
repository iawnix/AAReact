import numpy as np
from numpy.typing import NDArray

from typing import Any, List, Union, Tuple, Dict, Callable

# 绘制模型对不同数据集划分的敏感性
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from util.RegressMetrics import r2_score
from config.constants import RF_PARAM_GRID, XGB_PARAM_GRID, LGB_PARAM_GRID

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor

import pickle

def build_model(model_name: str, seed: int) -> Union[RandomForestRegressor, XGBRegressor, LGBMRegressor]:
    """
    构建模型
    """
    if model_name == "rf":
        return RandomForestRegressor(random_state=seed)
    elif model_name == "xgb":
        return XGBRegressor(random_state=seed)
    elif model_name == "lgb":
        return LGBMRegressor(random_state=seed, verbose=-1, silent=True)
    else:
        raise RuntimeError("Error[iaw]>: Unsupported model, {}".format(model_name))

def search_parms(model_name: str, X_train, y_train, seed: int, cv: int = 5) -> dict:
    """
    用于网格超参
    """
    if model_name == "rf":
        # grid search
        param_grid = RF_PARAM_GRID
    elif model_name == "xgb":
        param_grid = XGB_PARAM_GRID
    elif model_name == "lgb":
        param_grid = LGB_PARAM_GRID
    else:
        raise RuntimeError("Error[iaw]>: Unsupported model, {}".format(model_name))
    
    base_model = build_model(model_name, seed)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,                                  # k折
        scoring= "neg_mean_squared_error",      # 'neg_mean_squared_error',
        n_jobs=-1,
        verbose=0,
        return_train_score=True
    )
    grid_search.fit(X_train, y_train)
    best_idx = grid_search.best_index_
    train_score = -grid_search.cv_results_['mean_train_score'][best_idx]
    val_score = -grid_search.cv_results_['mean_test_score'][best_idx]
    print("Info[iaw]:> The trian mean mse: {:.4f}, the test mean mse: {:.4f}".format(train_score, val_score))
    return grid_search.best_params_

def eval_dataset_split(seed_s: List[int], test_size_s: List[int], parms: Dict, model_name: str
                        , data_x: NDArray, data_y: NDArray, data_class: List
                        , eval_func: callable) -> Tuple[list[float], list[float], list[float], list[float]]:
    """
    评估模型对数据集大小的依赖
    """
    train_score_mean_s, train_score_std_s = [], []
    test_score_mean_s, test_score_std_s = [], []
    for i_size in test_size_s:
        train_score_tmp = []
        test_score_tmp = []
        for i_seed in seed_s:
            _X_train, _X_test, _y_train, _y_test,  _class_train, _class_test = train_test_split(
                data_x,        
                data_y,
                data_class,
                test_size=i_size,
                random_state=i_seed, 
            )

            model = build_model(model_name, i_seed)
            model.set_params(**parms)
            model.fit(_X_train, _y_train)
            train_pred = model.predict(_X_train)
            test_pred = model.predict(_X_test)

            train_score_tmp.append(eval_func(y_pred = train_pred, y_true = _y_train ))
            test_score_tmp.append(eval_func(y_pred = test_pred, y_true = _y_test))
        train_score_mean_s.append(np.mean(train_score_tmp))
        train_score_std_s.append(np.std(train_score_tmp))
        test_score_mean_s.append(np.mean(test_score_tmp))
        test_score_std_s.append(np.std(test_score_tmp))

    return train_score_mean_s, train_score_std_s, test_score_mean_s, test_score_std_s

def load_data(data_x: str, data_y: str, x_label: str, data_class: str, data_name: Union[str, None] = None) -> Union[Tuple[NDArray, NDArray, List[str], List[int]], Tuple[NDArray, NDArray, List[str], List[int], List[str]]]:
    """
    加载数据
    """
    data_x = np.load("{}".format(data_x))
    data_y = np.load("{}".format(data_y))
    with open("{}".format(x_label), "rb") as f:
        x_label = pickle.load(f)
    with open("{}".format(data_class), "rb") as f:
        data_class = pickle.load(f)

    if data_name == None:
        return data_x, data_y, x_label, data_class
    else:
        with open(data_name, "rb") as f:
            data_name = pickle.load(f)
        return data_x, data_y, x_label, data_class, data_name

def split_data(data_s: Tuple[NDArray, NDArray, List[int]]
               , seed: int
               , test_size: float
               , data_name: Union[List[str], None] = None) -> Union[Tuple[NDArray, NDArray, NDArray, NDArray, List[int], List[int]], Tuple[NDArray, NDArray, NDArray, NDArray, List[int], List[int], List[str], List[str]]]:
    """
    划分训练测试数据
    20260327引入参数data_name, 并适配旧代码
    """
    if data_name == None:
        X_train, X_test, y_train, y_test,  class_train, class_test = train_test_split(
            data_s[0],        
            data_s[1],
            data_s[2],
            test_size=test_size,
            random_state=seed, 
        )
        return X_train, X_test, y_train, y_test, class_train, class_test
    
    else:
        X_train, X_test, y_train, y_test,  class_train, class_test, name_train, name_test = train_test_split(
            data_s[0],        
            data_s[1],
            data_s[2],
            data_name,
            test_size=test_size,
            random_state=seed, 
        )
        return X_train, X_test, y_train, y_test, class_train, class_test, name_train, name_test

    

def group_data(data_s: Tuple[NDArray, NDArray, List[str], List[int], List[str], List[int]]
               , group1: List[int]
               , group2: List[int]) -> Tuple[Tuple[NDArray, NDArray, List[str], List[int], List[str], List[int]], 
                                                 Tuple[NDArray, NDArray, List[str], List[int], List[str], List[int]]]:

    """
    基于batch划分数据集, 这个主要为了统一不同批次的数据
    """    
    data_x, data_y, x_label, data_class, data_name, data_batch = data_s
    
    group1_idx_s = []
    group2_idx_s = []
    for i, i_b in enumerate(data_batch):
        if i_b in group1:
            group1_idx_s.append(i)
        elif i_b in group2:
            group2_idx_s.append(i)
        else:
            pass

    return (
        (data_x[[group1_idx_s], :], data_y[[group1_idx_s]]
         , [i_t for i, i_t in enumerate(x_label) if i in group1_idx_s]
         , [i_t for i, i_t in enumerate(data_class) if i in group1_idx_s]
         , [i_t for i, i_t in enumerate(data_name) if i in group1_idx_s]
         , [i_t for i, i_t in enumerate(data_batch) if i in group1_idx_s]) , 
        (data_x[[group2_idx_s], :], data_y[[group2_idx_s]]
         , [i_t for i, i_t in enumerate(x_label) if i in group2_idx_s]
         , [i_t for i, i_t in enumerate(data_class) if i in group2_idx_s]
         , [i_t for i, i_t in enumerate(data_name) if i in group2_idx_s]
         , [i_t for i, i_t in enumerate(data_batch) if i in group2_idx_s])
    )

