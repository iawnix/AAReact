
from typing import Dict


def is_num(s:str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def touch_toml(out_fp: str,
                data_for_train_path: str
                , model_name: str
                , desc_type : str
                , batch_type: str
                , hypered_params: Dict[str, str]) -> None:
    with open(out_fp, "w+") as F:
        F.writelines("[Train]\n")
        F.writelines("data_x = \"{}/{}/{}_data_x.npy\"\n".format(data_for_train_path, desc_type, batch_type))
        F.writelines("data_y = \"{}/{}/{}_data_y.npy\"\n".format(data_for_train_path, desc_type, batch_type))
        F.writelines("x_label = \"{}/{}/{}_x_label.pkl\"\n".format(data_for_train_path, desc_type, batch_type))
        F.writelines("data_class = \"{}/{}/{}_data_class.pkl\"\n".format(data_for_train_path, desc_type, batch_type))
        F.writelines("seed = 1\n")
        F.writelines("test_size = 0.2\n")
        F.writelines("model_save = \"/home/iaw/DATA2/AAReact/train/output/pt/{}_{}_seed_0-1_test_0-2.pkl\"\n".format(model_name, desc_type))
        F.writelines("cv = 5\n")
        F.writelines("n_cpu = 5\n")
        F.writelines("\n")
        F.writelines("[Model]\n")
        F.writelines("model_type = \"{}\"\n".format(model_name))
        for key, item in hypered_params.items():
            if is_num(item):
                F.writelines("{} = {}\n".format(key, item))
            else:
                F.writelines("{} = \"{}\"\n".format(key, item))

def read_hyper_log(fp: str) -> Dict[str, str]:
    out = {}
    with open(fp, "r") as f:
        for line in f.readlines():
            _line = line.rstrip("\n")
            
            if _line == "Metric, TrainSet, TestSet":
                break
            
            if _line == "BestParams:":
                continue
            
            key, item = _line.split(":")
            out[key] = item.replace(" ", "")

    return out


model_s = ["lgb", "xgb", "rf"]
comb_s = [("rdkit", "soap"), ("soap", "xtb"), ("rdkit", "xtb"), ("soap", "acsf"), ("acsf", "xtb"), ("rdkit", "acsf")
               , ("rdkit", "soap", "xtb"), ("rdkit", "soap", "acsf"),("rdkit", "xtb", "acsf"), ("soap", "xtb", "acsf") 
               , ("rdkit", "soap", "xtb", "acsf")]
for i_m in model_s:
    for i_c in comb_s:
        i_desc_type = "_".join(i_c)
        log_f = "/home/iaw/DATA2/AAReact/train/output/hyper_log/{}_{}_seed_0-1_test_0-2_cv_5_hyper.log".format(i_m, i_desc_type)
        hyper_params = read_hyper_log(log_f)
        touch_toml(
              out_fp = "/home/iaw/DATA2/AAReact/config/train_ml_{}_{}.toml".format(i_m, i_desc_type)
            , data_for_train_path = "/home/iaw/DATA2/AAReact/DataSet/Data_All/4_data_for_train/"
            , model_name = i_m
            , desc_type = i_desc_type
            , batch_type = "train_test"
            , hypered_params = hyper_params
        )
