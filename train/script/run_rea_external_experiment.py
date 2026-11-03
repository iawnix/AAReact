#!/usr/bin/env python3
"""Train on REA-1 only and evaluate on fixed REA-2/3/4 external data."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump

PROJECT_ROOT = Path("/home/iaw/DATA2/AAReact")
SRC_DIR = PROJECT_ROOT / "src"
import sys

sys.path.append(str(SRC_DIR))

from config.constants import target_column
from util.RegressMetrics import mae_score, r2_score, rmse_score, r_score, spearmanr_score
from util.train_tools import build_model, search_params


DDG_R_KCAL = 0.001987
CELSIUS_TO_KELVIN = 273.15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "DataSet/Data_All3/3_data_for_train/ddg/rea1_train_rea234_test/rdkit_xtb_acsf",
    )
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=PROJECT_ROOT / "DataSet/Data_All3/full_data_rea1_train_rea234_test-20260922.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "train/output/Data_All3/rea1_train_rea234_test",
    )
    parser.add_argument("--target", choices=("ddg", "ee"), default="ddg")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--n-trials", type=int, default=400)
    parser.add_argument("--n-startup-trials", type=int, default=20)
    parser.add_argument("--n-cpu", type=int, default=5)
    return parser.parse_args()


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def ddg_to_ee(ddg: np.ndarray, temperature_c: np.ndarray) -> np.ndarray:
    temperature_k = np.asarray(temperature_c, dtype=float) + CELSIUS_TO_KELVIN
    return np.tanh(np.asarray(ddg, dtype=float) / (2.0 * DDG_R_KCAL * temperature_k))


def metric_payload(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    prediction = np.asarray(prediction, dtype=float).reshape(-1)
    zero_tol = 1.0e-12
    nonzero = np.abs(y_true) > zero_tol
    sign_correct = int(np.sum(np.sign(y_true[nonzero]) == np.sign(prediction[nonzero]))) if np.any(nonzero) else 0
    return {
        "n": int(len(y_true)),
        "rmse": float(rmse_score(prediction, y_true)),
        "mae": float(mae_score(prediction, y_true)),
        "r2": float(r2_score(prediction, y_true)),
        "pearson_r": float(r_score(prediction, y_true)),
        "spearman_r": float(spearmanr_score(prediction, y_true)),
        "true_zero_n": int(np.sum(~nonzero)),
        "sign_nonzero_n": int(np.sum(nonzero)),
        "sign_correct_n": sign_correct,
        "sign_accuracy_nonzero": float(sign_correct / np.sum(nonzero)) if np.any(nonzero) else float("nan"),
    }


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    train_x = np.load(data_dir / "train_data_x.npy")
    train_y = np.load(data_dir / "train_data_y.npy").reshape(-1)
    test_x = np.load(data_dir / "test_data_x.npy")
    test_y = np.load(data_dir / "test_data_y.npy").reshape(-1)
    x_label = load_pickle(data_dir / "train_x_label.pkl")
    train_names = load_pickle(data_dir / "train_data_name.pkl")
    test_names = load_pickle(data_dir / "test_data_name.pkl")
    if train_x.ndim != 2 or test_x.ndim != 2 or train_x.shape[1] != test_x.shape[1]:
        raise ValueError(f"Train/test feature shape mismatch: {train_x.shape} vs {test_x.shape}")
    if not np.isfinite(train_x).all() or not np.isfinite(test_x).all():
        raise ValueError("Train/test feature matrix contains NaN or inf")
    if len(train_x) != len(train_y) or len(test_x) != len(test_y):
        raise ValueError("Feature/target row count mismatch")
    if len(set(train_names) & set(test_names)):
        raise ValueError("Train/test DATA_ID overlap")

    study_dir = args.output_dir / "optuna" / "study"
    best_params, search_summary = search_params(
        model_name="xgb",
        X_train=train_x,
        y_train=train_y,
        seed=args.seed,
        n_cpu_opt=args.n_cpu,
        n_cpu_model=args.n_cpu,
        cv=args.cv,
        method="optuna",
        metric="rmse",
        shuffle_cv=True,
        n_trials=args.n_trials,
        n_startup_trials=args.n_startup_trials,
        objective_std_penalty=0.1,
        train_gap_penalty=0.0,
        study_dir=str(study_dir),
    )

    model = build_model("xgb", args.seed, args.n_cpu)
    model.set_params(**best_params)
    model.fit(train_x, train_y)
    train_prediction = np.asarray(model.predict(train_x), dtype=float)
    test_prediction = np.asarray(model.predict(test_x), dtype=float)
    model.aa_target = args.target
    model.aa_target_column = target_column(args.target)
    model.aa_x_label = x_label

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "model_xgb_rdkit_xtb_acsf.pkl"
    dump(model, model_path)

    source = pd.read_csv(args.source_csv, low_memory=False)
    source["DATA_ID"] = source["DATA_ID"].astype(str)
    test_meta = source.set_index("DATA_ID").loc[test_names].reset_index()
    train_meta = source.set_index("DATA_ID").loc[train_names].reset_index()
    if args.target == "ddg":
        test_ee = ddg_to_ee(test_prediction, test_meta["TEMP"].to_numpy(float))
    else:
        test_ee = np.full(len(test_prediction), np.nan)

    prediction_frame = test_meta[["DATA_ID", "REA_NAME", "CAT_NAME", "SOL_NAME", "TEMP", "PRESSURE", "BATCH"]].copy()
    prediction_frame["true_ddg"] = test_y if args.target == "ddg" else np.nan
    prediction_frame["pred_ddg"] = test_prediction if args.target == "ddg" else np.nan
    prediction_frame["error_pred_minus_true_ddg"] = test_prediction - test_y if args.target == "ddg" else np.nan
    prediction_frame["pred_ee"] = test_ee
    prediction_frame.to_csv(args.output_dir / "predictions.csv", index=False)

    metrics = {
        "train": metric_payload(train_y, train_prediction),
        "test_pooled": metric_payload(test_y, test_prediction),
    }
    by_rea = {}
    for rea in ("REA-2", "REA-3", "REA-4"):
        mask = test_meta["REA_NAME"].to_numpy() == rea
        by_rea[rea] = metric_payload(test_y[mask], test_prediction[mask])
    metrics["test_by_rea"] = by_rea

    write_json(args.output_dir / "metrics.json", metrics)
    write_json(args.output_dir / "optuna_best_params.json", best_params)
    write_json(args.output_dir / "optuna_search_summary.json", search_summary)
    manifest = {
        "target": args.target,
        "model": "xgb",
        "descriptor": "rdkit_xtb_acsf",
        "train_rea": "REA-1",
        "test_rea": ["REA-2", "REA-3", "REA-4"],
        "batch": 0,
        "seed": args.seed,
        "cv": args.cv,
        "n_trials": args.n_trials,
        "n_startup_trials": args.n_startup_trials,
        "train_rows": int(len(train_x)),
        "test_rows": int(len(test_x)),
        "features": int(train_x.shape[1]),
        "data_dir": str(data_dir),
        "source_csv": str(args.source_csv),
        "model_path": str(model_path),
        "prediction_path": str(args.output_dir / "predictions.csv"),
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps({"model": str(model_path), "predictions": str(args.output_dir / "predictions.csv"), "metrics": metrics, "features": int(train_x.shape[1])}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
