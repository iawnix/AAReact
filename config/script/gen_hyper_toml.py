import argparse
from pathlib import Path


CONFIG_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CONFIG_ROOT.parent
HYPER_LOG_DIR = PROJECT_ROOT / "hyper" / "output" / "hyper_log"
OPTUNA_STUDY_ROOT = PROJECT_ROOT / "hyper" / "output" / "optuna_study"


def touch_toml(
    out_fp: str,
    data_for_train_path: str,
    model_name: str,
    desc_type: str,
    batch_type: str,
    target: str,
    seed: int,
    test_size: float,
    split_name: str,
    search_method: str,
    cv: int,
    n_trials: int,
    n_startup_trials: int,
    objective_std_penalty: float,
    train_gap_penalty: float,
):
    target_suffix = "" if target == "ee" else "_{}".format(target)
    study_dir = ""
    if search_method == "optuna":
        study_dir = str(OPTUNA_STUDY_ROOT / target / split_name / "{}_{}".format(model_name, desc_type))
    with open(out_fp, "w+") as F:
        F.writelines("[Hyper]\n")
        F.writelines("target = \"{}\"\n".format(target))
        F.writelines("data_x = \"{}/{}/{}_data_x.npy\"\n".format(data_for_train_path, desc_type, batch_type))
        F.writelines("data_y = \"{}/{}/{}_data_y.npy\"\n".format(data_for_train_path, desc_type, batch_type))
        F.writelines("x_label = \"{}/{}/{}_x_label.pkl\"\n".format(data_for_train_path, desc_type, batch_type))
        F.writelines("data_class = \"{}/{}/{}_data_class.pkl\"\n".format(data_for_train_path, desc_type, batch_type))
        F.writelines("seed = {}\n".format(seed))
        F.writelines("test_size = {}\n".format(test_size))
        F.writelines("cv = {}\n".format(cv))
        F.writelines("n_cpu = 5\n")
        F.writelines("\n")
        F.writelines("[Search]\n")
        F.writelines("method = \"{}\"\n".format(search_method))
        F.writelines("metric = \"rmse\"\n")
        F.writelines("cv = {}\n".format(cv))
        F.writelines("shuffle_cv = true\n")
        F.writelines("n_trials = {}\n".format(n_trials))
        F.writelines("n_startup_trials = {}\n".format(n_startup_trials))
        F.writelines("objective_std_penalty = {}\n".format(objective_std_penalty))
        F.writelines("train_gap_penalty = {}\n".format(train_gap_penalty))
        F.writelines("study_dir = \"{}\"\n".format(study_dir))
        F.writelines("\n")
        F.writelines("[Model]\n")
        F.writelines("name = \"{}\"\n".format(model_name))
        F.writelines("n_cpu = 5\n")
        F.writelines("params_save = \"{}\"\n".format(
            HYPER_LOG_DIR / search_method / "{}_{}{}_{}_cv_{}_hyper.log".format(
                model_name, desc_type, target_suffix, split_name, cv
            )
        ))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ML hyperparameter-search TOML files.")
    parser.add_argument("--target", choices=["ee", "ddg"], default="ee")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--split-name", default="seed_1_test_0-2")
    parser.add_argument("--batch-type", default="train_test")
    parser.add_argument("--search-method", choices=["grid", "optuna"], default="grid")
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--n-trials", type=int, default=400)
    parser.add_argument("--n-startup-trials", type=int, default=20)
    parser.add_argument("--objective-std-penalty", type=float, default=0.1)
    parser.add_argument("--train-gap-penalty", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_for_train_path = "/home/iaw/DATA2/AAReact/DataSet/Data_All/3_data_for_train/{}/{}".format(args.target, args.split_name)
    out_dir = CONFIG_ROOT / "hyper" / args.target / args.split_name / args.search_method
    out_dir.mkdir(parents=True, exist_ok=True)
    model_s = ["lgb", "xgb", "rf"]
    comb_s = [("rdkit", "soap"), ("soap", "xtb"), ("rdkit", "xtb"), ("soap", "acsf"), ("acsf", "xtb"), ("rdkit", "acsf")
                   , ("rdkit", "soap", "xtb"), ("rdkit", "soap", "acsf"),("rdkit", "xtb", "acsf"), ("soap", "xtb", "acsf")
                   , ("rdkit", "soap", "xtb", "acsf"), ("rdkit", ), ("xtb", ), ("soap", ), ("acsf", )]
    for i_m in model_s:
        for i_c in comb_s:
            if len(i_c) == 1:
                i_desc_type = i_c[0]
            else:
                i_desc_type = "_".join(i_c)
            touch_toml(
                  out_fp = out_dir / "hyper_ml_{}_{}.toml".format(i_m, i_desc_type)
                , data_for_train_path = data_for_train_path
                , model_name = i_m
                , desc_type = i_desc_type
                , batch_type = args.batch_type
                , target = args.target
                , seed = args.seed
                , test_size = args.test_size
                , split_name = args.split_name
                , search_method = args.search_method
                , cv = args.cv
                , n_trials = args.n_trials
                , n_startup_trials = args.n_startup_trials
                , objective_std_penalty = args.objective_std_penalty
                , train_gap_penalty = args.train_gap_penalty
            )


if __name__ == "__main__":
    main()
