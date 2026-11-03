#!/usr/bin/env python3
"""Run one fixed-split Data_All4 training experiment.

The existing AHO_train entry point re-splits ``train_test`` internally. This
small runner keeps Data_All4's explicit train/test/extra membership intact,
while reusing the model parameters produced by the existing hyper workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import tomllib
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_FILES = {
    "rdkit": "rdkit_desc_features.csv",
    "xtb": "xtb_features.csv",
    "soap": "soap_features.csv",
    "acsf": "acsf_features.csv",
}
ALWAYS_KEEP = {"TEMP", "PRESSURE"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate one Data_All4 fixed-split experiment.")
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "DataSet" / "Data_All4")
    parser.add_argument("--split", default="seed_1_test_0-2")
    parser.add_argument("--target", default="ddg", choices=["ddg"], help="Data_All4 round 1 currently uses DDG.")
    parser.add_argument("--model", default="xgb", choices=["xgb", "lgb", "rf"])
    parser.add_argument("--descriptor", default="rdkit_xtb_acsf")
    parser.add_argument("--hyper-split", default="seed_1_test_0-2")
    parser.add_argument("--search-method", default="optuna", choices=["grid", "optuna"])
    parser.add_argument("--train-config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--skip-pool", action="store_true", help="Skip prediction for the 830-row unlabeled pool.")
    return parser.parse_args()


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def flatten_labels(payload: dict[str, list[str]]) -> list[str]:
    labels: list[str] = []
    for values in payload.values():
        labels.extend(values)
    return labels


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "n": int(len(y_true)),
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def load_split_part(dataset_dir: Path, prefix: str) -> dict[str, Any]:
    x = np.load(dataset_dir / f"{prefix}_data_x.npy")
    y = np.load(dataset_dir / f"{prefix}_data_y.npy")
    names = [str(value) for value in load_pickle(dataset_dir / f"{prefix}_data_name.pkl")]
    batches = [int(value) for value in load_pickle(dataset_dir / f"{prefix}_data_batch.pkl")]
    labels = load_pickle(dataset_dir / f"{prefix}_x_label.pkl")
    if x.shape[0] != len(y) or x.shape[0] != len(names) or x.shape[0] != len(batches):
        raise ValueError(f"{prefix} row counts do not match in {dataset_dir}")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f"{prefix} contains non-finite values in {dataset_dir}")
    return {"x": x, "y": y, "names": names, "batches": batches, "labels": labels}


def default_train_config(args: argparse.Namespace) -> Path:
    return (
        PROJECT_ROOT
        / "config"
        / "train"
        / args.target
        / f"hyper_{args.hyper_split}"
        / f"search_{args.search_method}"
        / args.split
        / f"train_ml_{args.model}_{args.descriptor}.toml"
    )


def build_model(config: dict[str, Any], project_root: Path):
    sys.path.insert(0, str(project_root / "src"))
    from util.train_tools import build_model as make_model

    train_cfg = config["Train"]
    model_cfg = dict(config["Model"])
    model_name = str(model_cfg.pop("model_type"))
    model = make_model(model_name, int(train_cfg.get("seed", 1)), int(train_cfg.get("n_cpu", 1)))
    model.set_params(**model_cfg)
    return model, model_name, int(train_cfg.get("seed", 1)), int(train_cfg.get("n_cpu", 1))


def save_predictions(path: Path, split: str, names: list[str], batches: list[int], y_true: np.ndarray | None, y_pred: np.ndarray) -> None:
    frame = pd.DataFrame({
        "DATA_ID": names,
        "BATCH": batches,
        "split": split,
        "pred_DDG": y_pred,
    })
    if y_true is not None:
        frame["true_DDG"] = y_true
        frame["error"] = y_pred - y_true
        frame["abs_error"] = np.abs(y_pred - y_true)
    frame.to_csv(path, index=False)


def build_pool_matrix(dataset_root: Path, target: str, split: str, descriptor: str, pool_ids: list[str]) -> tuple[np.ndarray, list[str]]:
    root = dataset_root / "3_data_for_train" / target / split
    parts: list[np.ndarray] = []
    labels: list[str] = []
    for index, feature in enumerate(descriptor.split("_")):
        if feature not in RAW_FILES:
            raise ValueError(f"Unsupported descriptor component: {feature}")
        raw = pd.read_csv(dataset_root / "2_raw_features" / RAW_FILES[feature], low_memory=False).set_index("DATA_ID")
        selector = json.loads((root / feature / "selector.json").read_text())
        selected = list(selector["selected_features"])
        if index > 0:
            selected = [label for label in selected if label not in ALWAYS_KEEP]
        missing = sorted(set(pool_ids) - set(raw.index))
        if missing:
            raise ValueError(f"Pool IDs missing from {feature} raw features: {missing[:5]}")
        values = raw.loc[pool_ids, selected].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Pool contains non-finite selected {feature} features")
        parts.append(values)
        labels.extend(selected)
    return np.concatenate(parts, axis=1), labels


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    dataset_dir = dataset_root / "3_data_for_train" / args.target / args.split / args.descriptor
    manifest_path = dataset_root / "manifest.json"
    if not dataset_dir.exists():
        raise FileNotFoundError(dataset_dir)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    train_config_path = (args.train_config or default_train_config(args)).resolve()
    if not train_config_path.exists():
        raise FileNotFoundError(train_config_path)
    config = tomllib.loads(train_config_path.read_text())

    manifest = json.loads(manifest_path.read_text())
    train = load_split_part(dataset_dir, "train")
    test = load_split_part(dataset_dir, "test")
    extra = load_split_part(dataset_dir, "extra")
    if train["names"] != [str(value) for value in manifest["train_ids"]]:
        raise ValueError("Data_All4 train IDs differ from manifest")
    if test["names"] != [str(value) for value in manifest["test_ids"]]:
        raise ValueError("Data_All4 test IDs differ from manifest")
    if extra["names"] != [str(value) for value in manifest["extra_ids"]]:
        raise ValueError("Data_All4 extra IDs differ from manifest")
    if train["labels"] != test["labels"] or train["labels"] != extra["labels"]:
        raise ValueError("Data_All4 train/test/extra feature labels differ")
    if set(train["names"]) & set(test["names"]) or set(train["names"]) & set(extra["names"]) or set(test["names"]) & set(extra["names"]):
        raise ValueError("Data_All4 train/test/extra IDs overlap")

    output_dir = (args.output_dir or PROJECT_ROOT / "train" / "output" / "Data_All4" / f"{args.model}_{args.descriptor}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model, model_name, seed, n_cpu = build_model(config, PROJECT_ROOT)
    model.fit(train["x"], train["y"])
    train_pred = np.asarray(model.predict(train["x"]), dtype=float)
    test_pred = np.asarray(model.predict(test["x"]), dtype=float)
    extra_pred = np.asarray(model.predict(extra["x"]), dtype=float)

    model.aa_target = args.target
    model.aa_target_column = "DDG" if args.target == "ddg" else "EE"
    model.aa_x_label = flatten_labels(train["labels"])
    joblib.dump(model, output_dir / f"model_{args.model}_{args.descriptor}.pkl")
    save_predictions(output_dir / "train_predictions.csv", "train", train["names"], train["batches"], train["y"], train_pred)
    save_predictions(output_dir / "test_predictions.csv", "test", test["names"], test["batches"], test["y"], test_pred)
    save_predictions(output_dir / "extra_predictions.csv", "extra", extra["names"], extra["batches"], extra["y"], extra_pred)

    merged = pd.read_csv(dataset_root / "full_data_merged_20261102.csv", low_memory=False)
    pool = merged.loc[merged["DDG"].isna()].copy()
    pool["DATA_ID"] = pool["DATA_ID"].astype(str)
    used_ids = set(train["names"]) | set(test["names"]) | set(extra["names"])
    pool = pool.loc[~pool["DATA_ID"].isin(used_ids)].copy()
    if set(pool["BATCH"].astype(int)) != {2} or len(pool) != 830:
        raise ValueError(f"Expected 830 unlabeled BATCH=2 pool rows, got {len(pool)}")
    pool_names = pool["DATA_ID"].tolist()
    pool_pred: np.ndarray | None = None
    if not args.skip_pool:
        pool_x, pool_labels = build_pool_matrix(dataset_root, args.target, args.split, args.descriptor, pool_names)
        if pool_labels != flatten_labels(train["labels"]):
            raise ValueError("Pool feature labels do not match Data_All4 training labels")
        pool_pred = np.asarray(model.predict(pool_x), dtype=float)
        pool_out = pd.DataFrame({
            "DATA_ID": pool_names,
            "BATCH": pool["BATCH"].astype(int).tolist(),
            "pred_DDG": pool_pred,
            "abs_pred_DDG": np.abs(pool_pred),
        }).sort_values("abs_pred_DDG", ascending=False)
        pool_out.to_csv(output_dir / "pool_predictions.csv", index=False)

    metrics_payload = {
        "train": metrics(train["y"], train_pred),
        "test": metrics(test["y"], test_pred),
        "extra": metrics(extra["y"], extra_pred),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2) + "\n")
    run_manifest = {
        "version": 1,
        "dataset": "Data_All4",
        "round": 1,
        "dataset_root": str(dataset_root),
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": sha256(manifest_path),
        "target": args.target,
        "split": args.split,
        "model": model_name,
        "descriptor": args.descriptor,
        "train_config": str(train_config_path),
        "hyper_split": args.hyper_split,
        "search_method": args.search_method,
        "seed": seed,
        "n_cpu": n_cpu,
        "row_counts": {"train": len(train["names"]), "test": len(test["names"]), "extra": len(extra["names"]), "pool": len(pool_names)},
        "pool_predictions_written": pool_pred is not None,
        "metrics": metrics_payload,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output_dir": str(output_dir), "row_counts": run_manifest["row_counts"], "metrics": metrics_payload}, indent=2))


if __name__ == "__main__":
    main()
