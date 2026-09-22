#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict


CONFIG_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CONFIG_ROOT.parent
HYPER_LOG_DIR = PROJECT_ROOT / "hyper" / "output" / "hyper_log"
TRAIN_CONFIG_ROOT = CONFIG_ROOT / "train"
TRAIN_MODEL_ROOT = PROJECT_ROOT / "train" / "output" / "pt"

MODEL_NAMES = ("lgb", "xgb", "rf")
DESCRIPTOR_NAMES = (
    "rdkit_soap",
    "soap_xtb",
    "rdkit_xtb",
    "soap_acsf",
    "acsf_xtb",
    "rdkit_acsf",
    "rdkit_soap_xtb",
    "rdkit_soap_acsf",
    "rdkit_xtb_acsf",
    "soap_xtb_acsf",
    "rdkit_soap_xtb_acsf",
    "rdkit",
    "xtb",
    "soap",
    "acsf",
)


def is_num(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def target_suffix(target: str) -> str:
    return "" if target == "ee" else "_{}".format(target)


def train_config_dir(target: str, hyper_split_name: str, train_split_name: str, search_method: str) -> Path:
    return TRAIN_CONFIG_ROOT / target / "hyper_{}".format(hyper_split_name) / "search_{}".format(search_method) / train_split_name


def train_model_dir(target: str, hyper_split_name: str, train_split_name: str, search_method: str) -> Path:
    return TRAIN_MODEL_ROOT / target / "hyper_{}".format(hyper_split_name) / "search_{}".format(search_method) / train_split_name


def hyper_log_path(model_name: str, desc_type: str, target: str, hyper_split_name: str, cv: int, search_method: str) -> Path:
    return HYPER_LOG_DIR / search_method / "{}_{}{}_{}_cv_{}_hyper.log".format(
        model_name,
        desc_type,
        target_suffix(target),
        hyper_split_name,
        cv,
    )


def read_hyper_log(fp: Path) -> Dict[str, str]:
    if not fp.exists():
        raise FileNotFoundError("Hyper log does not exist: {}".format(fp))

    out: Dict[str, str] = {}
    in_params = False
    with open(fp, "r") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            if text == "BestParams:":
                in_params = True
                continue
            if text == "Metric, TrainSet, TestSet":
                break
            if not in_params:
                continue
            key, value = text.split(":", 1)
            out[key.strip()] = value.strip()

    if not out:
        raise ValueError("No BestParams found in {}".format(fp))
    return out


def write_value(fp, key: str, value: str) -> None:
    text = str(value).strip()
    lower = text.lower()
    if lower == "true":
        fp.write("{} = true\n".format(key))
    elif lower == "false":
        fp.write("{} = false\n".format(key))
    elif lower in ("none", "null"):
        fp.write("{} = \"None\"\n".format(key))
    elif is_num(text):
        fp.write("{} = {}\n".format(key, text))
    else:
        fp.write("{} = \"{}\"\n".format(key, text))


def touch_toml(
    out_fp: Path,
    data_for_train_path: Path,
    model_save_dir: Path,
    model_name: str,
    desc_type: str,
    batch_type: str,
    hypered_params: Dict[str, str],
    target: str,
    seed: int,
    test_size: float,
    hyper_split_name: str,
    train_split_name: str,
    search_method: str,
    n_cpu: int,
) -> None:
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    model_save_dir.mkdir(parents=True, exist_ok=True)
    model_save = model_save_dir / "{}_{}.pkl".format(model_name, desc_type)

    with open(out_fp, "w") as f:
        f.write("# hyper_split_name = \"{}\"\n".format(hyper_split_name))
        f.write("# search_method = \"{}\"\n".format(search_method))
        f.write("# train_split_name = \"{}\"\n".format(train_split_name))
        f.write("[Train]\n")
        f.write("target = \"{}\"\n".format(target))
        f.write("data_x = \"{}/{}/{}_data_x.npy\"\n".format(data_for_train_path, desc_type, batch_type))
        f.write("data_y = \"{}/{}/{}_data_y.npy\"\n".format(data_for_train_path, desc_type, batch_type))
        f.write("x_label = \"{}/{}/{}_x_label.pkl\"\n".format(data_for_train_path, desc_type, batch_type))
        f.write("data_class = \"{}/{}/{}_data_class.pkl\"\n".format(data_for_train_path, desc_type, batch_type))
        f.write("seed = {}\n".format(seed))
        f.write("test_size = {}\n".format(test_size))
        f.write("model_save = \"{}\"\n".format(model_save))
        f.write("n_cpu = {}\n".format(n_cpu))
        f.write("\n")
        f.write("[Model]\n")
        f.write("model_type = \"{}\"\n".format(model_name))
        for key, value in hypered_params.items():
            write_value(f, key, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ML training TOML files from fixed hyperparameter logs.")
    parser.add_argument("--target", choices=["ee", "ddg"], default="ddg")
    parser.add_argument("--hyper-split-name", default="seed_1_test_0-2")
    parser.add_argument("--search-method", choices=["grid", "optuna"], default="grid")
    parser.add_argument("--train-split-name", default="seed_1_test_0-2")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--batch-type", default="train_test")
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--n-cpu", type=int, default=5)
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=list(MODEL_NAMES))
    parser.add_argument("--descriptors", nargs="+", default=list(DESCRIPTOR_NAMES), choices=list(DESCRIPTOR_NAMES))
    parser.add_argument("--list-outputs", action="store_true", help="Print every generated TOML path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_for_train_path = PROJECT_ROOT / "DataSet" / "Data_All" / "3_data_for_train" / args.target / args.train_split_name
    if not data_for_train_path.exists():
        raise FileNotFoundError("Dataset split does not exist: {}".format(data_for_train_path))

    out_dir = train_config_dir(args.target, args.hyper_split_name, args.train_split_name, args.search_method)
    model_save_dir = train_model_dir(args.target, args.hyper_split_name, args.train_split_name, args.search_method)
    outputs = []

    for model_name in args.models:
        for desc_type in args.descriptors:
            log_fp = hyper_log_path(model_name, desc_type, args.target, args.hyper_split_name, args.cv, args.search_method)
            if not log_fp.exists():
                legacy_fp = HYPER_LOG_DIR / log_fp.name
                if legacy_fp.exists():
                    log_fp = legacy_fp
            hyper_params = read_hyper_log(log_fp)
            out_fp = out_dir / "train_ml_{}_{}.toml".format(model_name, desc_type)
            touch_toml(
                out_fp=out_fp,
                data_for_train_path=data_for_train_path,
                model_save_dir=model_save_dir,
                model_name=model_name,
                desc_type=desc_type,
                batch_type=args.batch_type,
                hypered_params=hyper_params,
                target=args.target,
                seed=args.seed,
                test_size=args.test_size,
                hyper_split_name=args.hyper_split_name,
                train_split_name=args.train_split_name,
                search_method=args.search_method,
                n_cpu=args.n_cpu,
            )
            outputs.append(str(out_fp))

    summary = {
        "target": args.target,
        "hyper_split": args.hyper_split_name,
        "search_method": args.search_method,
        "train_split": args.train_split_name,
        "output_dir": str(out_dir),
        "model_save_dir": str(model_save_dir),
        "count": len(outputs),
    }
    if outputs:
        summary["first_output"] = outputs[0]
    if args.list_outputs:
        summary["outputs"] = outputs
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
