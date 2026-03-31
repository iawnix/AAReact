from typing import Union, List, Tuple
from numpy.typing import NDArray
import numpy as np
from rich import print as rprint
from rich.table import Table

def r_score(y_pred: Union[List, NDArray], y_true: Union[List, NDArray]) -> np.float64:
    """
    pearson correlation coefficient
    """
    if type(y_pred) == list:
        y_pred = np.array(y_pred)
    if type(y_true) == list:
        y_true = np.array(y_true)

    _dim1_true = y_true.shape[0]
    _dim1_pred = y_pred.shape[0]
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    assert y_pred.shape[0] == _dim1_pred and y_true.shape[0] == _dim1_true, "Error[iaw]>: This score if for single regression task."
    
    # 均值计算
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    # 协方差
    numerator = np.sum((y_true - mean_true) * (y_pred - mean_pred))
    
    # 标准差
    denominator = np.sqrt(np.sum((y_true - mean_true) ** 2)) * np.sqrt(np.sum((y_pred - mean_pred) ** 2))
    
    # calc
    if denominator == 0:
        return np.float64(0.0) 
    
    r = numerator / denominator
    return np.float64(r)

def spearmanr_score(y_pred: Union[List, NDArray], y_true: Union[List, NDArray]) -> np.float64:
    """
    spearmanr correlation coefficient
    """

    if type(y_pred) == list:
        y_pred = np.array(y_pred)
    if type(y_true) == list:
        y_true = np.array(y_true)

    _dim1_true = y_true.shape[0]
    _dim1_pred = y_pred.shape[0]
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    assert y_pred.shape[0] == _dim1_pred and y_true.shape[0] == _dim1_true, "Error[iaw]>: This score if for single regression task."

    # 计算秩
    rank_true = np.argsort(np.argsort(y_true))
    rank_pred = np.argsort(np.argsort(y_pred))

    # 对秩做 Pearson
    mean_t = np.mean(rank_true)
    mean_p = np.mean(rank_pred)

    numerator = np.sum((rank_true - mean_t) * (rank_pred - mean_p))
    denominator = np.sqrt(np.sum((rank_true - mean_t)**2)) * np.sqrt(np.sum((rank_pred - mean_p)**2))

    if denominator == 0:
        return np.float64(0.0)

    rho = numerator / denominator
    return np.float64(rho)

def r2_score(y_pred: Union[List, NDArray], y_true: Union[List, NDArray]) -> np.float64:

    if type(y_pred) == list:
        y_pred = np.array(y_pred)
    if type(y_true) == list:
        y_true = np.array(y_true)

    _dim1_true = y_true.shape[0]
    _dim1_pred = y_pred.shape[0]
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    assert y_pred.shape[0] == _dim1_pred and y_true.shape[0] == _dim1_true, "Error[iaw]>: This score if for single regression task."

    mean_y = np.mean(y_true)

    # 计算总平方和 SS_tot
    ss_tot = np.sum((y_true - mean_y) ** 2)

    # 计算残差平方和 SS_res
    ss_res = np.sum((y_true - y_pred) ** 2)
    # 计算 R²
    r2 = 1 - (ss_res / ss_tot)

    return r2

def mse_score(y_pred: Union[List, NDArray], y_true: Union[List, NDArray]) -> np.float64:
    if type(y_pred) == list:
        y_pred = np.array(y_pred)
    if type(y_true) == list:
        y_true = np.array(y_true)

    _dim1_true = y_true.shape[0]
    _dim1_pred = y_pred.shape[0]
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    assert y_pred.shape[0] == _dim1_pred and y_true.shape[0] == _dim1_true, "Error[iaw]>: This score if for single regression task."

    return  np.mean((y_true - y_pred) ** 2)

def mae_score(y_pred: Union[List, NDArray], y_true: Union[List, NDArray]) -> np.float64:
    if type(y_pred) == list:
        y_pred = np.array(y_pred)
    if type(y_true) == list:
        y_true = np.array(y_true)

    _dim1_true = y_true.shape[0]
    _dim1_pred = y_pred.shape[0]
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    assert y_pred.shape[0] == _dim1_pred and y_true.shape[0] == _dim1_true, "Error[iaw]>: This score if for single regression task."

    return np.sum(np.abs(y_true - y_pred))/y_true.shape[0]

def rmse_score(y_pred: Union[List, NDArray], y_true: Union[List, NDArray]) -> np.float64:
    if type(y_pred) == list:
        y_pred = np.array(y_pred)
    if type(y_true) == list:
        y_true = np.array(y_true)

    _dim1_true = y_true.shape[0]
    _dim1_pred = y_pred.shape[0]
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    assert y_pred.shape[0] == _dim1_pred and y_true.shape[0] == _dim1_true, "Error[iaw]>: This score if for single regression task."

    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def print_metric(name: List[str], data_s: List[Tuple[float, float, float]]) ->None:
    from rich import print as rp
    """
    用于输出评分指标的表格, 输入name和数据, 数据必须是一个包含(Train, Valid, Test)的列表
    """
    for i in data_s:
        if len(i) != 3:
            raise RuntimeError("Error[iaw]>: please must privide (Train, Valid, Test).")
    table = Table(
        title="[green bold]Metric[/green bold]",  
        show_header=True,
        header_style="bold yellow",  # 表头样式：加粗+蓝色
        border_style="white",        # 表格边框颜色
        title_justify="center",      # 标题居中
        width=60                     # 表格宽度（可根据需要调整）
    )

    table.add_column("name", justify="center", style="white")
    table.add_column("Train", justify="center", style="white")
    table.add_column("Valid", justify="center", style="white")
    table.add_column("Test", justify="center", style="white")
    for metric_name, metric_data in zip(name, data_s):
        table.add_row(metric_name, "{:.4f}".format(metric_data[0]), "{:.4f}".format(metric_data[1]), "{:.4f}".format(metric_data[2]))
    rp(table)


if __name__ == "__main__":

    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    rprint({"shape": y_true.shape, "y_true":y_true})
    rprint("MSE:", mse_score(y_pred, y_true))
    rprint("MAE:", mae_score(y_pred, y_true))
    rprint("R²:", r2_score(y_pred, y_true))
    rprint("RMSE:", rmse_score(y_pred, y_true))

    y_true = np.array([[3],[-0.5] ,[2] ,[7]])
    y_pred = np.array([[2.5],[0.0] ,[2] ,[8]])
    rprint({"shape": y_true.shape, "y_true":y_true})
    rprint("MSE:", mse_score(y_pred, y_true))
    rprint("MAE:", mae_score(y_pred, y_true))
    rprint("R²:", r2_score(y_pred, y_true))
    rprint("RMSE:", rmse_score(y_pred, y_true))

    y_true = np.array([[3, -0.5, 2, 7], [3, -0.5, 2, 7]])
    y_pred = np.array([[2.5, 0.0, 2, 8], [2.5, 0.0, 2, 8]])
    rprint({"shape": y_true.shape, "y_true":y_true})
    rprint("MSE:", mse_score(y_pred, y_true))
    rprint("MAE:", mae_score(y_pred, y_true))
    rprint("R²:", r2_score(y_pred, y_true))
    rprint("RMSE:", rmse_score(y_pred, y_true))