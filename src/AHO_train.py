import os
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from util.RegressMetrics import r2_score, mse_score, mae_score, rmse_score
from util.train_tools import build_model, search_parms, split_data, load_data
from config.ml_train import init_config_from_train_toml
from config.ml_hyper import init_config_from_hyper_toml
from util.RegressMetrics import print_metric

import argparse
from argparse import Namespace

from joblib import dump

from rich.status import Status
from rich import print as rp
from dataclasses import asdict

def Parm() -> Namespace:
    """
    cli 参数
    """
    parser = argparse.ArgumentParser(description="AAReact: Atropic Acid Enantioselectivity Prediction")
    parser.add_argument("--task", type=str, help="Indicate whether the task is `hyperparameter tuning`[hyper] or pure `model training`[train].")
    parser.add_argument("--model_config", type=str, help="Model training config file in TOML format.")
    
    args = parser.parse_args()
    return args


def main() -> None:
    
    myp = Parm()
    if myp.task == "train":
        with Status("Train model...", spinner = "pong") as status:
            # init config

            config_fp = myp.model_config
            trian_config = init_config_from_train_toml(config_fp) 

            # split data
            data_x, data_y, x_label, data_class = load_data(
                trian_config.Train.data_x, 
                trian_config.Train.data_y, 
                trian_config.Train.x_label, 
                trian_config.Train.data_class
            )
            #print("Debug[iaw]:> data_x.shape = {}, data_y.shape = {}, len(x_label) = {}, len(data_class) = {}".format(
            #    data_x.shape, data_y.shape, len(x_label), len(data_class)
            #))
            X_train, X_test, y_train, y_test, class_train, class_test = split_data(data_s = (data_x, data_y, data_class)
                       , seed = trian_config.Train.seed
                       , test_size = trian_config.Train.test_size)

            # train model
            #print("Debug[iaw]:> trian_config.Model: {}".format(asdict(trian_config.Model)))
            model = build_model(trian_config.Model_type, trian_config.Train.seed)
            model.set_params(**asdict(trian_config.Model))
            model.fit(X_train, y_train)

        # assessment
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        train_r2 = r2_score(y_pred=train_pred, y_true=y_train)
        test_r2 = r2_score(y_pred=test_pred, y_true=y_test)
        train_mse = mse_score(y_pred=train_pred, y_true=y_train)
        test_mse = mse_score(y_pred=test_pred, y_true=y_test)
        train_mae = mae_score(y_pred=train_pred, y_true=y_train)
        test_mae = mae_score(y_pred=test_pred, y_true=y_test)
        train_rmse = rmse_score(y_pred=train_pred, y_true=y_train)
        test_rmse = rmse_score(y_pred=test_pred, y_true=y_test)
        print_metric(["R2", "MSE", "MAE", "RMSE"], [(train_r2, 0, test_r2), (train_mse, 0, test_mse), (train_mae, 0, test_mae), (train_rmse, 0, test_rmse)])

        # save 
        rp("Info\\[iaw]:> The model will be saved as `{}`".format(trian_config.Train.model_save))
        dump(model, trian_config.Train.model_save)
    elif myp.task == "hyper":
        config_fp = myp.model_config
        hyper_config = init_config_from_hyper_toml(config_fp) 
        print("Info[iaw]:> Start hyper model: {}".format(hyper_config.Model_type))
        with Status("Hyper model...", spinner = "pong") as status:
            # split data
            data_x, data_y, x_label, data_class = load_data(
                hyper_config.Hyper.data_x, 
                hyper_config.Hyper.data_y, 
                hyper_config.Hyper.x_label, 
                hyper_config.Hyper.data_class
            )
            #print("Debug[iaw]:> data_x.shape = {}, data_y.shape = {}, len(x_label) = {}, len(data_class) = {}".format(
            #    data_x.shape, data_y.shape, len(x_label), len(data_class)
            #))
            X_train, X_test, y_train, y_test, class_train, class_test = split_data(data_s = (data_x, data_y, data_class)
                       , seed = hyper_config.Hyper.seed
                       , test_size = hyper_config.Hyper.test_size)
            best_params = search_parms(
                model_name = hyper_config.Model_type, 
                X_train = X_train, 
                y_train = y_train,
                seed = hyper_config.Hyper.seed, 
                cv = hyper_config.Hyper.cv)
            print("Info[iaw]>: {} best params: {}".format(hyper_config.Model_type, best_params))
            # best model
            save_model = build_model(hyper_config.Model_type, hyper_config.Hyper.seed)
            save_model.set_params(**best_params)
            save_model.fit(X_train, y_train)
            train_pred = save_model.predict(X_train)
            test_pred = save_model.predict(X_test)
            # metrics
            with open(hyper_config.params_save, "w+") as F:
                F.writelines("BestParams:\n")
                for k, v in best_params.items():
                    F.writelines("{}: {}\n".format(k, v))
                F.writelines("Metric, TrainSet, TestSet\n")
                F.writelines("R2, {:.4F}, {:.4F}\n".format(
                    r2_score(y_pred=train_pred, y_true=y_train), r2_score(y_pred=test_pred, y_true=y_test)
                ))
                F.writelines("MSE, {:.4F}, {:.4F}\n".format(
                    mse_score(y_pred=train_pred, y_true=y_train), mse_score(y_pred=test_pred, y_true=y_test)
                ))
                F.writelines("MAE, {:.4F}, {:.4F}\n".format(
                    mae_score(y_pred=train_pred, y_true=y_train), mae_score(y_pred=test_pred, y_true=y_test)
                ))
                F.writelines("RMSE, {:.4F}, {:.4F}\n".format(
                    rmse_score(y_pred=train_pred, y_true=y_train), rmse_score(y_pred=test_pred, y_true=y_test)
                ))
    else:
        print("Error\\[iaw]:> The task is `hyperparameter tuning`[hyper] or pure `model training`[train]!")
        

if __name__ == "__main__":
    main()