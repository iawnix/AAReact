import pandas as pd
import numpy as np
from numpy.typing import NDArray

from matplotlib import pyplot as plt

from typing import Any, List, Union, Tuple, Dict, Callable

from rich.progress import track

# 绘制模型对不同数据集划分的敏感性
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from util.RegressMetrics import r2_score


def get_desc_type_split_line(x_label: list[str]) -> dict[str, int]:
    """
    获取不同描述符的分割线
    """
    out = {}
    for i, i_txt in enumerate(x_label):
        if (i_type:=i_txt.split("_")[0]) not in out.keys():
            out[i_type] = []
        out[i_type].append(i)
    for key in dict(out).keys():
        i_values = sorted(out[key])
        out[key] = i_values[-1]
    return out

def draw_pearson_corr(data_x: NDArray, data_y: NDArray, x_label: list[str]):
    """
    对相关性进行可视化
    """
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

def draw_data_x_distribution(data_x: NDArray, x_label: list[str]):
    """
    绘制特征的数据分布
    """
    plt.figure(figsize=(12, 7), dpi = 300)
    ax = plt.gca()
    plt.imshow(data_x, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.xlabel("Feature")
    plt.ylabel("Sample")
    
    type_split = [v+0.2 for k, v in get_desc_type_split_line(x_label).items()]
    ax.set_xticks(type_split)
    ax.set_xticklabels([])

def draw_data_x_var_distribution(data_x: NDArray, x_label: list[str]):
    """
    绘制特征的方差分布
    """
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

def draw_pred_result(y_true_s: Tuple[NDArray, NDArray], y_pred_s: Tuple[NDArray, NDArray], r2_s: Tuple[float, float], class_s: Tuple[List, List]) -> None:   
    """
    用于预测结果的展示
    """
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
