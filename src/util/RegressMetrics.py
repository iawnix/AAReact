from typing import Union, List
from numpy.typing import NDArray
import numpy as np
from rich import print as rprint

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