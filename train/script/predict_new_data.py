#!/usr/bin/env python3
"""Predict AAReact targets for prepared ML feature datasets.

The expected input is the feature matrix produced by the DataSet workflow,
for example:

  DataSet/Data_All/3_data_for_train/ddg/seed_1_test_0-2/rdkit_soap_acsf/extra_data_x.npy

For raw molecule inputs, use src/AHO_predict.py instead.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import load


TRAIN_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAIN_DIR.parent
DDG_R_KCAL = 0.001987
CELSIUS_TO_KELVIN = 273.15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict ee/ddG for prepared AAReact ML feature datasets."
    )
    parser.add_argument("--target", choices=["ee", "ddg"], required=True)
    parser.add_argument("--hyper-split", required=True, help="Hyperparameter split name, e.g. seed_1_test_0-2.")
    parser.add_argument("--search-method", choices=["grid", "optuna"], required=True)
    parser.add_argument("--split", required=True, help="Prediction data split name, e.g. seed_1_test_0-15.")
    parser.add_argument("--model-name", choices=["lgb", "xgb", "rf"], required=True)
    parser.add_argument("--descriptor", required=True, help="Descriptor directory name, e.g. rdkit_xtb.")
    parser.add_argument("--dataset", required=True, help="Dataset prefix, e.g. extra or train_test.")
    parser.add_argument("--model", type=Path, default=None, help="Explicit model .pkl path.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing <dataset>_data_x.npy and related files.",
    )
    parser.add_argument("--data-x", type=Path, default=None, help="Explicit feature matrix .npy path.")
    parser.add_argument("--data-y", type=Path, default=None, help="Optional target .npy path.")
    parser.add_argument("--x-label", type=Path, default=None, help="Optional x_label .pkl path.")
    parser.add_argument("--data-name", type=Path, default=None, help="Optional data_name .pkl path.")
    parser.add_argument("--data-class", type=Path, default=None, help="Optional data_class .pkl path.")
    parser.add_argument("--data-batch", type=Path, default=None, help="Optional data_batch .pkl path.")
    parser.add_argument("--output", type=Path, default=None, help="Output CSV path.")
    parser.add_argument(
        "--temperature-c",
        type=float,
        default=None,
        help="Fallback Celsius temperature for ddG to ee conversion when TEMP is not present in x_label.",
    )
    parser.add_argument(
        "--max-abs-value",
        type=float,
        default=1.0e12,
        help="Fail if abs(feature) is larger than this value.",
    )
    parser.add_argument(
        "--allow-label-mismatch",
        action="store_true",
        help="Do not fail when model feature labels differ from input x_label.",
    )
    return parser.parse_args()


def safe_name(text: str) -> str:
    return str(text).replace(".", "-").replace("/", "_")


def inferred_data_dir(args: argparse.Namespace) -> Path:
    return (
        PROJECT_ROOT
        / "DataSet"
        / "Data_All"
        / "3_data_for_train"
        / args.target
        / args.split
        / args.descriptor
    )


def inferred_model_path(args: argparse.Namespace) -> Path:
    return (
        TRAIN_DIR
        / "output"
        / "pt"
        / args.target
        / f"hyper_{args.hyper_split}"
        / f"search_{args.search_method}"
        / args.split
        / f"{args.model_name}_{args.descriptor}.pkl"
    )


def inferred_output_path(args: argparse.Namespace) -> Path:
    name = "{}_hyper_{}_search_{}_split_{}_{}_{}_{}_predictions.csv".format(
        args.target,
        safe_name(args.hyper_split),
        args.search_method,
        safe_name(args.split),
        args.model_name,
        args.descriptor,
        args.dataset,
    )
    return TRAIN_DIR / "predict_new" / name


def existing_optional(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.exists() else None


def paths_from_args(args: argparse.Namespace) -> dict[str, Path | None]:
    data_dir = args.data_dir or inferred_data_dir(args)
    dataset = args.dataset
    paths = {
        "model": args.model or inferred_model_path(args),
        "data_x": args.data_x or data_dir / f"{dataset}_data_x.npy",
        "data_y": args.data_y or existing_optional(data_dir / f"{dataset}_data_y.npy"),
        "x_label": args.x_label or existing_optional(data_dir / f"{dataset}_x_label.pkl"),
        "data_name": args.data_name or existing_optional(data_dir / f"{dataset}_data_name.pkl"),
        "data_class": args.data_class or existing_optional(data_dir / f"{dataset}_data_class.pkl"),
        "data_batch": args.data_batch or existing_optional(data_dir / f"{dataset}_data_batch.pkl"),
        "output": args.output or inferred_output_path(args),
    }
    return paths


def require_file(path: Path | None, label: str) -> Path:
    if path is None or not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def load_pickle(path: Path | None) -> Any | None:
    if path is None:
        return None
    with path.open("rb") as f:
        return pickle.load(f)


def load_model(path: Path) -> tuple[Any, str | None, Any | None]:
    obj = load(path)
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"], obj.get("target"), obj.get("x_label")
    return obj, getattr(obj, "aa_target", None), getattr(obj, "aa_x_label", None)


def label_key_order(key: str) -> tuple[int, str]:
    match = re.match(r"label(\d+)_", str(key))
    if match:
        return int(match.group(1)), str(key)
    return 999999, str(key)


def flatten_labels(labels: Any | None) -> list[str] | None:
    if labels is None:
        return None
    if isinstance(labels, dict):
        out: list[str] = []
        for key in sorted(labels, key=label_key_order):
            out.extend([str(item) for item in labels[key]])
        return out
    if isinstance(labels, (list, tuple)):
        return [str(item) for item in labels]
    raise TypeError(f"Unsupported x_label type: {type(labels)}")


def ddg_to_ee(ddg: np.ndarray, temp_c: np.ndarray) -> np.ndarray:
    temp_k = np.asarray(temp_c, dtype=float) + CELSIUS_TO_KELVIN
    if np.any(temp_k <= 0):
        raise ValueError("Temperature in Kelvin must be positive for ddG to ee conversion.")
    return np.tanh(np.asarray(ddg, dtype=float) / (2.0 * DDG_R_KCAL * temp_k))


def resolve_temperature_c(
    data_x: np.ndarray,
    input_labels: Any | None,
    fallback_temperature_c: float | None,
) -> np.ndarray | None:
    input_flat = flatten_labels(input_labels)
    if input_flat is not None and "TEMP" in input_flat:
        return data_x[:, input_flat.index("TEMP")]
    if fallback_temperature_c is not None:
        return np.full(data_x.shape[0], float(fallback_temperature_c), dtype=float)
    return None


def validate_feature_matrix(data_x: np.ndarray, max_abs_value: float) -> np.ndarray:
    if data_x.ndim == 1:
        data_x = data_x.reshape(1, -1)
    if data_x.ndim != 2:
        raise ValueError(f"data_x must be a 2D matrix, got shape {data_x.shape}")

    data_x = np.asarray(data_x, dtype=np.float64)
    invalid = ~np.isfinite(data_x)
    if invalid.any():
        rows, cols = np.where(invalid)
        examples = list(zip(rows[:5].tolist(), cols[:5].tolist()))
        raise ValueError(
            "Feature matrix contains NaN/inf values: count={}, examples={}".format(
                int(invalid.sum()), examples
            )
        )

    if data_x.size:
        max_abs = float(np.max(np.abs(data_x)))
        if max_abs > max_abs_value:
            row, col = np.unravel_index(np.argmax(np.abs(data_x)), data_x.shape)
            raise ValueError(
                "Feature matrix contains too-large values: max_abs={:.6g} at row={}, col={}, "
                "threshold={:.6g}".format(max_abs, int(row), int(col), max_abs_value)
            )
    return data_x


def validate_feature_contract(
    model: Any,
    data_x: np.ndarray,
    input_labels: Any | None,
    model_labels: Any | None,
    allow_label_mismatch: bool,
) -> None:
    expected_n = getattr(model, "n_features_in_", None)
    if expected_n is not None and int(expected_n) != data_x.shape[1]:
        raise ValueError(
            "Feature dimension mismatch: model expects {}, input has {}".format(
                int(expected_n), data_x.shape[1]
            )
        )

    input_flat = flatten_labels(input_labels)
    model_flat = flatten_labels(model_labels)

    if input_flat is not None and len(input_flat) != data_x.shape[1]:
        raise ValueError(
            "x_label length mismatch: x_label has {}, data_x has {}".format(
                len(input_flat), data_x.shape[1]
            )
        )

    if model_flat is not None and len(model_flat) != data_x.shape[1]:
        raise ValueError(
            "model aa_x_label length mismatch: aa_x_label has {}, data_x has {}".format(
                len(model_flat), data_x.shape[1]
            )
        )

    if input_flat is None or model_flat is None:
        return

    if input_flat != model_flat and not allow_label_mismatch:
        first_mismatch = next(
            (
                idx
                for idx, (left, right) in enumerate(zip(input_flat, model_flat))
                if left != right
            ),
            None,
        )
        if first_mismatch is None and len(input_flat) != len(model_flat):
            first_mismatch = min(len(input_flat), len(model_flat))
        raise ValueError(
            "Feature labels differ between input and model at index {}. "
            "Use --allow-label-mismatch only if feature order is known to be compatible.".format(
                first_mismatch
            )
        )


def sequence_or_default(values: Any | None, n_rows: int, default_prefix: str) -> list[Any]:
    if values is None:
        return [f"{default_prefix}_{idx}" for idx in range(n_rows)]
    if len(values) != n_rows:
        raise ValueError(f"Metadata length mismatch: expected {n_rows}, got {len(values)}")
    return list(values)


def build_output_frame(
    args: argparse.Namespace,
    data_x: np.ndarray,
    pred: np.ndarray,
    true_y: np.ndarray | None,
    temperature_c: np.ndarray | None,
    names: Any | None,
    classes: Any | None,
    batches: Any | None,
) -> pd.DataFrame:
    n_rows = data_x.shape[0]
    target = args.target.lower()
    out = pd.DataFrame(
        {
            "row_index": np.arange(n_rows, dtype=int),
            "data_name": sequence_or_default(names, n_rows, "row"),
            "target": target,
            "model": args.model_name,
            "descriptor": args.descriptor,
            "hyper_split": args.hyper_split,
            "search_method": args.search_method,
            "split": args.split,
            "dataset": args.dataset,
            f"pred_{target}": pred,
        }
    )
    if true_y is not None:
        if true_y.ndim != 1:
            true_y = np.ravel(true_y)
        if len(true_y) != n_rows:
            raise ValueError(f"data_y length mismatch: expected {n_rows}, got {len(true_y)}")
        out[f"true_{target}"] = true_y
        out[f"error_pred_minus_true_{target}"] = pred - true_y
    if target == "ddg" and temperature_c is not None:
        if len(temperature_c) != n_rows:
            raise ValueError(f"temperature length mismatch: expected {n_rows}, got {len(temperature_c)}")
        if true_y is not None:
            out["true_ee"] = ddg_to_ee(true_y, temperature_c)
        out["pred_ee"] = ddg_to_ee(pred, temperature_c)
    if classes is not None:
        out["data_class"] = sequence_or_default(classes, n_rows, "class")
    if batches is not None:
        out["data_batch"] = sequence_or_default(batches, n_rows, "batch")
    return out


def main() -> None:
    args = parse_args()
    paths = paths_from_args(args)

    model_path = require_file(paths["model"], "model")
    data_x_path = require_file(paths["data_x"], "data_x")

    model, model_target, model_labels = load_model(model_path)
    if model_target is not None and str(model_target).lower() != args.target.lower():
        raise ValueError(
            "Model target metadata ({}) does not match --target ({})".format(
                model_target, args.target
            )
        )

    data_x = validate_feature_matrix(np.load(data_x_path), args.max_abs_value)
    data_y = np.load(paths["data_y"]) if paths["data_y"] is not None else None
    input_labels = load_pickle(paths["x_label"])
    names = load_pickle(paths["data_name"])
    classes = load_pickle(paths["data_class"])
    batches = load_pickle(paths["data_batch"])

    validate_feature_contract(
        model=model,
        data_x=data_x,
        input_labels=input_labels,
        model_labels=model_labels,
        allow_label_mismatch=args.allow_label_mismatch,
    )
    temperature_c = resolve_temperature_c(data_x, input_labels, args.temperature_c)

    pred = np.asarray(model.predict(data_x), dtype=float).reshape(-1)
    if len(pred) != data_x.shape[0]:
        raise ValueError("Prediction length mismatch: expected {}, got {}".format(data_x.shape[0], len(pred)))

    out_df = build_output_frame(
        args=args,
        data_x=data_x,
        pred=pred,
        true_y=data_y,
        temperature_c=temperature_c,
        names=names,
        classes=classes,
        batches=batches,
    )

    output_path = paths["output"]
    if output_path is None:
        raise RuntimeError("Internal error: output path was not resolved.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    print(
        json.dumps(
            {
                "model": str(model_path),
                "data_x": str(data_x_path),
                "data_y": str(paths["data_y"]) if paths["data_y"] is not None else None,
                "output": str(output_path),
                "rows": int(data_x.shape[0]),
                "features": int(data_x.shape[1]),
                "target": args.target,
                "prediction_min": float(np.min(pred)) if len(pred) else None,
                "prediction_max": float(np.max(pred)) if len(pred) else None,
                "prediction_mean": float(np.mean(pred)) if len(pred) else None,
                "ee_columns_added": bool(args.target == "ddg" and temperature_c is not None),
                "temperature_c_min": float(np.min(temperature_c)) if temperature_c is not None else None,
                "temperature_c_max": float(np.max(temperature_c)) if temperature_c is not None else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
