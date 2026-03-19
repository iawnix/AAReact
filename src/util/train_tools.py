import pandas as pd
import numpy as np
from numpy.typing import NDArray

from matplotlib import pyplot as plt

from typing import Any, List, Union, Tuple, Dict, Callable


from rich.table import Table
from rich import print as rp
from rich.progress import track

# 绘制模型对不同数据集划分的敏感性
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from util.RegressMetrics import r2_score


# 用于加载csv数据集, 并删除nan值
def load_data(data_fp: str, desc_type: str):
    # load
    data = pd.read_csv(data_fp)
    
    # drop nan and inf
    print("Infor[iaw]>: raw data set shape: {}".format(data.shape))
    data = data.dropna(axis=1)
    print("Infor[iaw]>: after del nan, data set shape: {}".format(data.shape))
    
    col_n = data.columns.to_list()
    ee_idx = col_n.index('EE')

    del_n = []
    if desc_type == "rdkit_desc":
        del_n.append("CAT_Ipc")
    for i_n in col_n[ee_idx+1:-1]:
        if np.isinf(data.loc[:, i_n].to_numpy()).any(axis = 0):
            del_n.append(i_n)
    for i_n in del_n:
        del data[i_n]
    print("Infor[iaw]>: after del inf, data set shape: {}".format(data.shape))

    # 开始分割数据
    # split -> data_x, data_y, x_label, data_class
    col_n = data.columns.to_list()
    ee_idx = col_n.index('EE')
    temp_idx = col_n.index('TEMP')
    pressure_idx = col_n.index('PRESSURE')
    class_idx = col_n.index('CLASS')
    assert class_idx == len(col_n)-1, "Error[iaw]>: class_idx != len(col_n)-1"
    # 不能拼接CLASS
    data_x = np.concat([data.iloc[:, ee_idx+1:-1].values, data.iloc[:,[temp_idx, pressure_idx]].values], axis = 1)
    data_y = data.iloc[:, ee_idx].values
    x_label = data.iloc[:, ee_idx+1:-1].columns.to_list() + data.iloc[:,[temp_idx, pressure_idx]].columns.to_list()
    data_class = data.iloc[:, class_idx].to_list()
    assert data_x.shape[-1] == len(x_label), "Error[iaw]>: data_x.shape[-1] != len(x_label)"

    return data_x, data_y, x_label, data_class

# 删除数据中标准差为0的特征
def std_zero_filter(data_x: NDArray, x_label: list[str]) -> tuple[NDArray, list[str]]:
    print("Infor[iaw]>: before del zero std, data_x shape: {}, x_label shape: {}".format(data_x.shape, len(x_label)))
    del_zero_std_idxs = []
    for i, i_txt in enumerate(x_label):
        _x = data_x[:, i]
        if np.isclose(np.std(_x, axis=0), 0, atol=1e-8):
            #print("Warning[iaw]>: feature {} has zero std.".format(i_txt))
            del_zero_std_idxs.append(i)
    data_x = np.delete(data_x, del_zero_std_idxs, axis=1)
    x_label = [i for j, i in enumerate(x_label) if j not in del_zero_std_idxs]
    print("Infor[iaw]>: after del zero std, data_x shape: {}, x_label shape: {}".format(data_x.shape, len(x_label)))
    return data_x, x_label

# 相关性筛选, 会去除与预测值相关性小于threshold的特征, 返回筛选后的data_x和x_label
def pearson_corr_filter(data_x: NDArray, data_y: NDArray, x_label: list[str], threshold: float = 0.05) -> tuple[NDArray, list[str]]:
    pear_result = []
    # 拼接
    pear = np.corrcoef(np.hstack([data_x, data_y.reshape(-1, 1)]).T)
    pear_y = pear[:, -1]    # -1是EE
    del_low_pear_idxs = []

    for i, i_txt in enumerate(x_label):
        if abs(pear_y[i]) < threshold:
            del_low_pear_idxs.append(i)
        else:
            pear_result.append((i_txt, pear_y[i]))
    data_x = np.delete(data_x, del_low_pear_idxs, axis=1)
    x_label = [i for j, i in enumerate(x_label) if j not in del_low_pear_idxs]
    print("Infor[iaw]>: after del low pearson corr, data_x shape: {}, x_label shape: {}".format(data_x.shape, len(x_label)))

    # 防止顺序错乱, 检查保留的特征与x_label是否一致
    com_list = [i[0] for i in pear_result]
    for i, i_txt in enumerate(x_label):
        if com_list[i] != i_txt:
            print("Error[iaw]>: pear_result and x_label not match at index {}.".format(i))

    return data_x, x_label    

# 对相关性进行可视化
def draw_pearson_corr(data_x: NDArray, data_y: NDArray, x_label: list[str]):
    
    # 计算相关性
    pear = np.corrcoef(np.hstack([data_x, data_y.reshape(-1, 1)]).T)
    pear_y = pear[:, -1]
    colors = ['red' if c > 0 else 'blue' for c in pear_y]  # 正相关红，负相关蓝
    
    plt.figure(figsize=(30, 8), dpi=300)
    bars = plt.bar(x_label+["EE"], pear_y, color=colors, alpha=0.7)
    type_split = [v for k, v in get_desc_type_split_line(x_label).items()]
    
    for i in type_split:
        i_x = [i+0.2, i+0.2]
        i_y = [-1, 1]
        plt.plot(i_x, i_y, color='black', linewidth=1)
    
    
    ax = plt.gca()
    ax.set_xticks(type_split)
    ax.set_xticklabels([])
   
    plt.ylim([-1, 1])
    plt.xlabel("Feature")
    plt.ylabel("Pearson Correlation")

# 对特征进行归一化, 并返回特征归一化的时候的最大值与最小值, 以便后续对测试集进行同样的归一化
def norm_col(x: NDArray) -> NDArray:
    min_ = np.min(x, axis=0)
    max_ = np.max(x, axis=0)
    return (x - min_) / (max_ - min_ + 1e-8), min_, max_

# 对特征进行归一化, 这个类需要加载norm_col得到的最大值和最小值, 以便对测试集进行同样的归一化
class norm_col_parms():
    def __init__(self, min_col: Tuple[NDArray, Any] = None, max_col: Tuple[NDArray, Any] = None):
        self.min_col = None
        self.max_col = None
        if min_col is not None and max_col is not None:
            self.set_params(min_col, max_col)

    def set_params(self, min_col: NDArray, max_col: NDArray):
        self.min_col = min_col
        self.max_col = max_col
    
    def __call__(self,x: NDArray) -> NDArray:
        return (x - self.min_col) / (self.max_col - self.min_col + 1e-8)


# 获取不同描述符的分割线
def get_desc_type_split_line(x_label: list[str]) -> dict[str, int]:
    out = {}
    for i, i_txt in enumerate(x_label):
        if (i_type:=i_txt.split("_")[0]) not in out.keys():
            out[i_type] = []
        out[i_type].append(i)
    for key in dict(out).keys():
        i_values = sorted(out[key])
        out[key] = i_values[-1]
    return out

# 绘制特征的数据分布
def draw_data_x_distribution(data_x: NDArray, x_label: list[str]):

    plt.figure(figsize=(12, 7), dpi = 300)
    ax = plt.gca()
    plt.imshow(data_x, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.xlabel("Feature")
    plt.ylabel("Sample")
    
    type_split = [v+0.2 for k, v in get_desc_type_split_line(x_label).items()]
    ax.set_xticks(type_split)
    ax.set_xticklabels([])

# 绘制特征的方差分布
def draw_data_x_var_distribution(data_x: NDArray, x_label: list[str]):
    plt.figure(figsize=(25, 4), dpi = 300)
    ax = plt.gca()
    plt.bar(range(len(np.var(data_x, axis = 0))), np.var(data_x, axis = 0), color='blue')
    plt.xlabel("Feature")
    plt.ylabel("Variance")

    type_split = [v+0.2 for k, v in get_desc_type_split_line(x_label).items()]
    for i in type_split:
        i_x = [i+0.2, i+0.2]
        i_y = [-1, 1]
        plt.plot(i_x, i_y, color='black', linewidth=1)

    ax.set_xticks(type_split)
    ax.set_xticklabels([])
    plt.ylim([0, 0.4])

# 用于输出评分指标的表格, 输入name和数据, 数据必须是一个包含(Train, Valid, Test)的列表
def print_metric(name: List[str], data_s:List[Tuple[float, float, float]]) ->None:
    for i in data_s:
        if len(i) != 3:
            raise RuntimeError("Error[iaw]>: please must privide (Train, Valid, Test).")
    table = Table(
        title="[green bold]Metric[/green bold]",  
        show_header=True,
        header_style="bold yellow",  # 表头样式：加粗+蓝色
        border_style="white",       # 表格边框颜色
        title_justify="center",    # 标题居中
        width=60                   # 表格宽度（可根据需要调整）
    )

    table.add_column("name", justify="center", style="white")
    table.add_column("Train", justify="center", style="white")
    table.add_column("Valid", justify="center", style="white")
    table.add_column("Test", justify="center", style="white")
    for metric_name, metric_data in zip(name, data_s):
        table.add_row(metric_name, "{:.4f}".format(metric_data[0]), "{:.4f}".format(metric_data[1]), "{:.4f}".format(metric_data[2]))
    rp(table)


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
import lightgbm as lgb

# 构建模型
def build_model(model_name: str, seed: int):
    if model_name == "rf":
        return RandomForestRegressor(random_state=seed)
    elif model_name == "xgb":
        return XGBRegressor(random_state=seed)
    elif model_name == "lgb":
        return lgb.LGBMRegressor(random_state=seed)
    else:
        raise RuntimeError("Error[iaw]>: Unsupported model, {}".format(model_name))

# 用于网格超参
def search_parms(model_name: str, X_train, y_train, seed: int) -> dict:
    if model_name == "rf":
        # grid search
        param_grid = {
            'n_estimators': [1, 3, 5, 10, 20, 30, 50, 70, 100, 120, 130, 140, 145, 150, 155, 160, 200], 
            'max_depth': [1, 3, 5, 8, 10, 20, 30, 40, 50, None], 
            'min_samples_split': [2, 5, 10, 30, 50],  
            'min_samples_leaf': [4, 5, 10, 30, 50], 
        }
    elif model_name == "xgb":
        param_grid = {
            'n_estimators': [1, 3, 5, 7, 10, 15, 20, 25, 30],  #1, 3, 5, 10, 20 , 30, 50, 70, 100, 120, 130, 140, 145, 150, 155, 160, 200, 太容易过拟合了
            'max_depth': [1, 3, 5, 8, 10, 20], # , 30, 40, 50
            "reg_alpha": [0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
            'reg_lambda': [0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
            'learning_rate': [0.01, 0.1],      
            'min_child_weight': [5, 10],         
            'subsample': [0.5, 0.6, 0.8],  
            'colsample_bytree': [0.8, 0.9],    
        }
    elif model_name == "lgb":
        param_grid = {
            'n_estimators': list(range(1, 100, 50)) + [120, 130, 140, 145, 150, 155, 160, 200],
            'max_depth': [1, 5, 10, 20, 30, 40, 50],
            'learning_rate': [0.01, 0.1, 0.2],      
            'num_leaves': [31, 50, 100],  
            'min_child_samples': [20, 50], 
            'subsample': [0.8, 0.9, 1.0], 
            'colsample_bytree': [0.8, 0.9, 1.0], 
        }
    else:
        raise RuntimeError("Error[iaw]>: Unsupported model, {}".format(model_name))
    
    base_model = build_model(model_name, seed)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,                                  # 5折
        scoring= "neg_mean_squared_error",     # 'neg_mean_squared_error',
        n_jobs=-1,
        verbose=0,
        return_train_score=True
    )
    grid_search.fit(X_train, y_train)
    best_idx = grid_search.best_index_
    train_score = -grid_search.cv_results_['mean_train_score'][best_idx]
    val_score = -grid_search.cv_results_['mean_test_score'][best_idx]
    print("Info[iaw]:> the trian mean mse: {:.4f}, the test mean mse: {:.4f}".format(train_score, val_score))
    return grid_search.best_params_

# 用于预测结果的展示
def draw_pred_result(y_true_s: Tuple[NDArray, NDArray], y_pred_s: Tuple[NDArray, NDArray], r2_s: Tuple[float, float], class_s: Tuple[List, List]) -> None:
    train_pred, test_pred = y_pred_s
    y_train, y_test = y_true_s
    train_r2, test_r2 = r2_s
    class_train, class_test = class_s   
    # draw scatter
    color_s = ["#06b6d4", "#8b5cf6", "#14b8a6", "#60f43f", "#d45a5a"]
    fig, ax = plt.subplots(1, 2, figsize = (10, 5), dpi = 300)
    ax[0].scatter(train_pred, y_train, c = [color_s[i] for i in class_train])
    ax[0].plot([-1, 1], [-1, 1], label = "Train, R2: {:.4f}".format(train_r2), c = "gray")
    ax[0].plot([0, 0], [-1, 1], c = "gray", linestyle = "--")
    ax[0].plot([-1, 1], [0, 0], c = "gray", linestyle = "--")
    ax[0].set_xlim([-1, 1])
    ax[0].set_ylim([-1, 1])
    ax[0].set_xlabel("Predicted")
    ax[0].set_ylabel("True")
    
    ax[1].scatter(test_pred, y_test, c = [color_s[i] for i in class_test])
    ax[1].plot([-1, 1], [-1, 1], label = "Test, R2: {:.4f}".format(test_r2), c = "gray")
    ax[1].plot([0, 0], [-1, 1], c = "gray", linestyle = "--")
    ax[1].plot([-1, 1], [0, 0], c = "gray", linestyle = "--")
    ax[1].set_xlim([-1, 1])
    ax[1].set_ylim([-1, 1])
    ax[1].set_xlabel("Predicted")
    ax[1].set_ylabel("True")
    
    ax[0].legend()
    ax[1].legend()


def eval_dataset_split(seed_s: List[int], test_size_s: List[int], parms: Dict, model_name: str
                        , data_x: NDArray, data_y: NDArray, data_class: List
                        , eval_func: callable) -> Tuple[list[float], list[float], list[float], list[float]]:
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

