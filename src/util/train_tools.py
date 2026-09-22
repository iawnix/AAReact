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
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_validate
from xgboost import XGBRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor

import json
import pickle
from datetime import datetime

def build_model(model_name: str, seed: int, n_cpu: int) -> Union[RandomForestRegressor, XGBRegressor, LGBMRegressor]:
    """
    构建模型
    """
    if model_name == "rf":
        return RandomForestRegressor(random_state=seed, n_jobs = n_cpu)
    elif model_name == "xgb":
        return XGBRegressor(random_state=seed, n_jobs= n_cpu)
    elif model_name == "lgb":
        return LGBMRegressor(
            random_state=seed,
            verbose=-1,
            silent=True,
            num_threads=n_cpu if n_cpu != -1 else 0,
            subsample_freq=1,
        )
    else:
        raise RuntimeError("Error[iaw]>: Unsupported model, {}".format(model_name))


def _param_grid_for(model_name: str) -> dict:
    if model_name == "rf":
        return RF_PARAM_GRID
    elif model_name == "xgb":
        return XGB_PARAM_GRID
    elif model_name == "lgb":
        return LGB_PARAM_GRID
    else:
        raise RuntimeError("Error[iaw]>: Unsupported model, {}".format(model_name))


def _scoring_for_metric(metric: str) -> str:
    metric = str(metric).lower()
    if metric == "rmse":
        return "neg_root_mean_squared_error"
    if metric == "mse":
        return "neg_mean_squared_error"
    raise ValueError("Unsupported search metric: {}. Use rmse or mse.".format(metric))


def _cv_splitter(cv: int, seed: int, shuffle_cv: bool):
    if int(cv) < 2:
        raise ValueError("cv must be >= 2, got {}".format(cv))
    if shuffle_cv:
        return KFold(n_splits=int(cv), shuffle=True, random_state=seed)
    return KFold(n_splits=int(cv), shuffle=False)


def _positive_score(values: NDArray, metric: str) -> NDArray:
    score = -np.asarray(values, dtype=float)
    if str(metric).lower() not in ("rmse", "mse"):
        raise ValueError("Unsupported search metric: {}".format(metric))
    return score


def _objective_value(mean_score: float, std_score: float, train_gap: float,
                     objective_std_penalty: float, train_gap_penalty: float) -> float:
    return (
        float(mean_score)
        + float(objective_std_penalty) * float(std_score)
        + float(train_gap_penalty) * max(float(train_gap), 0.0)
    )


def _suggest_optuna_params(trial, model_name: str) -> dict:
    if model_name == "rf":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
            "max_depth": trial.suggest_categorical(
                "max_depth",
                [None, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 30, 40],
            ),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 30),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
            "ccp_alpha": trial.suggest_float("ccp_alpha", 1e-8, 5e-2, log=True),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        }
    if model_name == "xgb":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 30, 400, step=10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        }
    if model_name == "lgb":
        max_depth = trial.suggest_int("max_depth", 2, 8)
        max_leaves = min(64, max(4, 2 ** max_depth - 1))
        return {
            "n_estimators": trial.suggest_int("n_estimators", 30, 400, step=10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": max_depth,
            "num_leaves": trial.suggest_int("num_leaves", 4, max_leaves),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 80),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 2.0),
        }
    raise RuntimeError("Error[iaw]>: Unsupported model, {}".format(model_name))


def _write_json(fp: Path, payload: dict) -> None:
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def search_params_grid(model_name: str, X_train, y_train, seed: int, n_cpu_opt: int, n_cpu_model: int,
                       cv: int = 5, metric: str = "rmse", shuffle_cv: bool = True,
                       objective_std_penalty: float = 0.1, train_gap_penalty: float = 0.0) -> tuple[dict, dict]:
    """
    GridSearchCV hyperparameter search.
    """
    param_grid = _param_grid_for(model_name)
    scoring = _scoring_for_metric(metric)
    cv_obj = _cv_splitter(cv, seed, shuffle_cv)
    base_model = build_model(model_name, seed, n_cpu = n_cpu_model)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv_obj,
        scoring=scoring,
        n_jobs= n_cpu_opt,
        verbose=0,
        return_train_score=True,
        refit=False,
        error_score="raise",
    )
    grid_search.fit(X_train, y_train)
    train_scores = _positive_score(grid_search.cv_results_['mean_train_score'], metric)
    val_scores = _positive_score(grid_search.cv_results_['mean_test_score'], metric)
    val_stds = np.asarray(grid_search.cv_results_['std_test_score'], dtype=float)
    train_gaps = val_scores - train_scores
    objectives = np.asarray([
        _objective_value(val_score, val_std, train_gap, objective_std_penalty, train_gap_penalty)
        for val_score, val_std, train_gap in zip(val_scores, val_stds, train_gaps)
    ])
    best_idx = int(np.argmin(objectives))
    train_score = float(train_scores[best_idx])
    val_score = float(val_scores[best_idx])
    val_std = float(val_stds[best_idx])
    train_gap = val_score - train_score
    objective = float(objectives[best_idx])
    print("Info[iaw]:> grid mean train {}: {:.4f}, mean cv {}: {:.4f}, std cv {}: {:.4f}".format(
        metric, train_score, metric, val_score, metric, val_std
    ))
    summary = {
        "method": "grid",
        "metric": metric,
        "cv": int(cv),
        "shuffle_cv": bool(shuffle_cv),
        "best_train_mean": train_score,
        "best_cv_mean": val_score,
        "best_cv_std": val_std,
        "best_train_cv_gap": train_gap,
        "objective_std_penalty": float(objective_std_penalty),
        "train_gap_penalty": float(train_gap_penalty),
        "best_objective": objective,
    }
    return dict(grid_search.cv_results_["params"][best_idx]), summary


def search_params_optuna(model_name: str, X_train, y_train, seed: int, n_cpu_opt: int, n_cpu_model: int,
                         cv: int = 5, metric: str = "rmse", shuffle_cv: bool = True,
                         n_trials: int = 150, n_startup_trials: int = 20,
                         objective_std_penalty: float = 0.1, train_gap_penalty: float = 0.0,
                         study_dir: Union[str, None] = None) -> tuple[dict, dict]:
    """
    Optuna hyperparameter search with CV inside each trial.
    """
    import optuna

    scoring = _scoring_for_metric(metric)
    cv_obj = _cv_splitter(cv, seed, shuffle_cv)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial) -> float:
        params = _suggest_optuna_params(trial, model_name)
        model = build_model(model_name, seed, n_cpu=n_cpu_model)
        model.set_params(**params)
        scores = cross_validate(
            model,
            X_train,
            y_train,
            scoring=scoring,
            cv=cv_obj,
            n_jobs=n_cpu_opt,
            return_train_score=True,
            error_score="raise",
        )
        cv_scores = _positive_score(scores["test_score"], metric)
        train_scores = _positive_score(scores["train_score"], metric)
        mean_cv = float(np.mean(cv_scores))
        std_cv = float(np.std(cv_scores))
        mean_train = float(np.mean(train_scores))
        train_gap = mean_cv - mean_train
        value = _objective_value(mean_cv, std_cv, train_gap, objective_std_penalty, train_gap_penalty)
        trial.set_user_attr("mean_cv_{}".format(metric), mean_cv)
        trial.set_user_attr("std_cv_{}".format(metric), std_cv)
        trial.set_user_attr("mean_train_{}".format(metric), mean_train)
        trial.set_user_attr("train_cv_gap_{}".format(metric), train_gap)
        return value

    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=int(n_startup_trials))
    storage = None
    study_name = None
    study_path = None
    if study_dir:
        study_path = Path(study_dir)
        study_path.mkdir(parents=True, exist_ok=True)
        storage = "sqlite:///{}".format(study_path / "study.db")
        study_name = "study_{}".format(datetime.now().strftime("%Y%m%d_%H%M%S"))

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        storage=storage,
        study_name=study_name,
        load_if_exists=False,
    )
    study.optimize(objective, n_trials=int(n_trials), n_jobs=1, show_progress_bar=False)

    best_trial = study.best_trial
    best_params = dict(best_trial.params)
    summary = {
        "method": "optuna",
        "metric": metric,
        "cv": int(cv),
        "shuffle_cv": bool(shuffle_cv),
        "n_trials": int(n_trials),
        "n_startup_trials": int(n_startup_trials),
        "objective_std_penalty": float(objective_std_penalty),
        "train_gap_penalty": float(train_gap_penalty),
        "best_objective": float(best_trial.value),
        "best_cv_mean": float(best_trial.user_attrs.get("mean_cv_{}".format(metric), np.nan)),
        "best_cv_std": float(best_trial.user_attrs.get("std_cv_{}".format(metric), np.nan)),
        "best_train_mean": float(best_trial.user_attrs.get("mean_train_{}".format(metric), np.nan)),
        "best_train_cv_gap": float(best_trial.user_attrs.get("train_cv_gap_{}".format(metric), np.nan)),
        "best_trial_number": int(best_trial.number),
    }

    if study_path is not None:
        trials_fp = study_path / "trials.csv"
        study.trials_dataframe().to_csv(trials_fp, index=False)
        _write_json(study_path / "best_params.json", best_params)
        _write_json(study_path / "search_summary.json", summary)
        summary["study_dir"] = str(study_path)
        summary["trials_csv"] = str(trials_fp)

    print("Info[iaw]:> optuna mean train {}: {:.4f}, mean cv {}: {:.4f}, std cv {}: {:.4f}".format(
        metric, summary["best_train_mean"], metric, summary["best_cv_mean"], metric, summary["best_cv_std"]
    ))
    return best_params, summary


def search_params(model_name: str, X_train, y_train, seed: int, n_cpu_opt: int, n_cpu_model: int,
                  cv: int = 5, method: str = "grid", metric: str = "rmse", shuffle_cv: bool = True,
                  n_trials: int = 150, n_startup_trials: int = 20,
                  objective_std_penalty: float = 0.1, train_gap_penalty: float = 0.0,
                  study_dir: Union[str, None] = None) -> tuple[dict, dict]:
    method = str(method).lower()
    if method == "grid":
        return search_params_grid(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            seed=seed,
            n_cpu_opt=n_cpu_opt,
            n_cpu_model=n_cpu_model,
            cv=cv,
            metric=metric,
            shuffle_cv=shuffle_cv,
            objective_std_penalty=objective_std_penalty,
            train_gap_penalty=train_gap_penalty,
        )
    if method == "optuna":
        return search_params_optuna(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            seed=seed,
            n_cpu_opt=n_cpu_opt,
            n_cpu_model=n_cpu_model,
            cv=cv,
            metric=metric,
            shuffle_cv=shuffle_cv,
            n_trials=n_trials,
            n_startup_trials=n_startup_trials,
            objective_std_penalty=objective_std_penalty,
            train_gap_penalty=train_gap_penalty,
            study_dir=study_dir,
        )
    raise ValueError("Unsupported search method: {}. Use grid or optuna.".format(method))


def search_parms(model_name: str, X_train, y_train, seed: int, n_cpu_opt: int, n_cpu_model: int, cv: int = 5) -> dict:
    """
    Backward-compatible wrapper for the historical GridSearchCV interface.
    """
    best_params, _summary = search_params(
        model_name=model_name,
        X_train=X_train,
        y_train=y_train,
        seed=seed,
        n_cpu_opt=n_cpu_opt,
        n_cpu_model=n_cpu_model,
        cv=cv,
        method="grid",
    )
    return best_params

def eval_dataset_split(seed_s: List[int], test_size_s: List[int], parms: Dict, model_name: str
                        , data_x: NDArray, data_y: NDArray, data_class: List
                       , eval_func: callable, n_cpu: int) -> Tuple[list[float], list[float], list[float], list[float], Dict[str, List]]:
    """
    评估模型对数据集大小的依赖
    """
    train_score_mean_s, train_score_std_s = [], []
    test_score_mean_s, test_score_std_s = [], []
    feature_importance_s = {}
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

            model = build_model(model_name, i_seed, n_cpu)
            model.set_params(**parms)
            model.fit(_X_train, _y_train)
            train_pred = model.predict(_X_train)
            test_pred = model.predict(_X_test)
            importances: NDArray = model.feature_importances_
            
            if str(i_seed) not in feature_importance_s.keys():
                feature_importance_s[str(i_seed)] = [importances]
            else:
                feature_importance_s[str(i_seed)].append(importances)

            train_score_tmp.append(eval_func(y_pred = train_pred, y_true = _y_train ))
            test_score_tmp.append(eval_func(y_pred = test_pred, y_true = _y_test))
        train_score_mean_s.append(np.mean(train_score_tmp))
        train_score_std_s.append(np.std(train_score_tmp))
        test_score_mean_s.append(np.mean(test_score_tmp))
        test_score_std_s.append(np.std(test_score_tmp))

    return train_score_mean_s, train_score_std_s, test_score_mean_s, test_score_std_s, feature_importance_s

def load_data(data_x: str, data_y: str, x_label: str, data_class: str, data_name: Union[str, None] = None, data_batch: Union[str, None] = None) -> List:
    """
    加载数据
    """

    data_x = np.load("{}".format(data_x))
    data_y = np.load("{}".format(data_y))
    with open("{}".format(x_label), "rb") as f:
        x_label = pickle.load(f)
    with open("{}".format(data_class), "rb") as f:
        data_class = pickle.load(f)
    out = [data_x, data_y, x_label, data_class]
    if data_name != None:
        with open(data_name, "rb") as f:
            data_name = pickle.load(f)
            out.append(data_name)
    if data_batch != None:
        with open(data_batch, "rb") as f:
            data_batch = pickle.load(f)
            out.append(data_batch)
    return out

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
    return:
        [0]: group1
            [0]: data_x
            [1]: data_y
            [2]: x_label
            [3]: data_class
            [4]: data_name
            [5]: data_batch
        [1]: group2
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
        (data_x[group1_idx_s, :], data_y[group1_idx_s]
         , x_label
         , [i_t for i, i_t in enumerate(data_class) if i in group1_idx_s]
         , [i_t for i, i_t in enumerate(data_name) if i in group1_idx_s]
         , [i_t for i, i_t in enumerate(data_batch) if i in group1_idx_s]) , 
        (data_x[group2_idx_s, :], data_y[group2_idx_s]
         , x_label
         , [i_t for i, i_t in enumerate(data_class) if i in group2_idx_s]
         , [i_t for i, i_t in enumerate(data_name) if i in group2_idx_s]
         , [i_t for i, i_t in enumerate(data_batch) if i in group2_idx_s])
    )
