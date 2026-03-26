import pickle

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

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

def load_raw_csv_data(data_fp: str, desc_type: str) -> Tuple[NDArray, NDArray, List[str], List[int]]:
    """
    用于加载csv数据集, 并删除nan值
    """
    # load
    data = pd.read_csv(data_fp)
    
    # drop nan and inf
    print("Infor[iaw]>: raw data set shape: {}".format(data.shape))
    data = data.dropna(axis=1)
    print("Infor[iaw]>: after del nan, data set shape: {}".format(data.shape))
    
    col_n = data.columns.to_list()
    ee_idx = col_n.index('EE')

    del_n = []
    # https://github.com/rdkit/rdkit/issues/1527
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

def load_raw_csv_data_base_label(data_fp: str, x_label: List[str]) -> Tuple[NDArray, NDArray, List[str], List[int]]:
    """
    用于新增数据的加载, 此处会基于x_label_fp种的label进行过滤
    """
    # load
    data = pd.read_csv(data_fp)
    
    col_n = data.columns.to_list()
    ee_idx = col_n.index('EE')

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
    x_label_out = data.iloc[:, ee_idx+1:-1].columns.to_list() + data.iloc[:,[temp_idx, pressure_idx]].columns.to_list()

    # 这里需要根据x_label对data_x进行删减
    select_x_label_idx_s = [x_label_out.index(label) for label in x_label]
    # data_x: n, n_feat
    data_x = data_x[:, select_x_label_idx_s]

    data_class = data.iloc[:, class_idx].to_list()
    assert data_x.shape[-1] == len(x_label), "Error[iaw]>: data_x.shape[-1] != len(x_label)"

    return data_x, data_y, x_label, data_class


def std_zero_filter(data_x: NDArray, x_label: List[str]) -> Tuple[NDArray, List[str]]:
    """
    删除数据中标准差为0的特征
    """
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


def pearson_corr_filter(data_x: NDArray, data_y: NDArray, x_label: List[str], threshold: float = 0.05) -> Tuple[NDArray, List[str], List[int]]:
    """
    相关性筛选, 会去除与预测值相关性小于threshold的特征, 返回筛选后的data_x和x_label, 这个筛选在特征工程的时候, 应该只用于Train, 以防止特征泄漏
    return: 特征筛选后的data_x, x_label以及选择特征的索引
    """
    
    pear_result = []
    # 拼接
    pear = np.corrcoef(np.hstack([data_x, data_y.reshape(-1, 1)]).T)
    pear_y = pear[:, -1]    # -1是EE
    del_low_pear_idxs = []
    select_idx_s = []
    # 这里不删除TEMP和PRESSURE
    for i, i_txt in enumerate(x_label):
        if abs(pear_y[i]) < threshold:
            if i_txt in ["TEMP", "PRESSURE"]:
                pear_result.append((i_txt, pear_y[i]))
                select_idx_s.append(i)
            else:
                del_low_pear_idxs.append(i)
        else:
            pear_result.append((i_txt, pear_y[i]))
            select_idx_s.append(i)
    data_x = np.delete(data_x, del_low_pear_idxs, axis=1)
    x_label = [i for j, i in enumerate(x_label) if j not in del_low_pear_idxs]
    print("Infor[iaw]>: after del low pearson corr, data_x shape: {}, x_label shape: {}".format(data_x.shape, len(x_label)))

    # 防止顺序错乱, 检查保留的特征与x_label是否一致
    com_list = [i[0] for i in pear_result]
    for i, i_txt in enumerate(x_label):
        if com_list[i] != i_txt:
            print("Error[iaw]>: pear_result and x_label not match at index {}.".format(i))

    return data_x, x_label, select_idx_s  

def f_regression_filter(data_x: NDArray, data_y: NDArray, x_label: list[str], k: int = 256) -> Tuple[NDArray, List[str], List[int]]:
    selector = SelectKBest(score_func = f_regression, k = k)
    selector.fit(data_x, data_y)
    select_idx_s = selector.get_support(indices=True)
    select_idx_s = list(select_idx_s)
    # 不删除温度和压强
    TEMP_idx = x_label.index("TEMP")
    PRESSURE_idx = x_label.index("PRESSURE")
    if TEMP_idx not in select_idx_s:
        select_idx_s.append(TEMP_idx)
    if PRESSURE_idx not in select_idx_s:
        select_idx_s.append(PRESSURE_idx)

    select_idx_s = sorted(select_idx_s, reverse=False) # 从小到大
    return data_x[:, select_idx_s], [x_label[i] for i in select_idx_s], select_idx_s

def norm_col(x: NDArray) -> NDArray:
    """
    对特征进行归一化, 并返回特征归一化的时候的最大值与最小值, 以便后续对测试集进行同样的归一化
    """
    min_ = np.min(x, axis=0)
    max_ = np.max(x, axis=0)
    return (x - min_) / (max_ - min_ + 1e-8), min_, max_

class norm_col_parms():
    """
    对特征进行归一化, 这个类需要加载norm_col得到的最大值和最小值, 以便对测试集进行同样的归一化
    """
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


