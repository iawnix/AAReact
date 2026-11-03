#!/usr/bin/env python3
"""Build a leakage-safe fixed-REA external test dataset for AAReact."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_regression


PROJECT_ROOT = Path("/home/iaw/DATA2/AAReact")
RAW_FILES = {
    "rdkit": "rdkit_desc_features.csv",
    "xtb": "xtb_features.csv",
    "acsf": "acsf_features.csv",
}
FEATURE_CONFIG = {
    "rdkit": {"k": 256, "drop": ("CAT_Ipc",)},
    "xtb": {"k": None, "drop": ()},
    "acsf": {"k": 128, "drop": ()},
}
ALWAYS_KEEP = ("TEMP", "PRESSURE")
MAX_ABS = float(np.finfo(np.float32).max)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        type=Path,
        default=PROJECT_ROOT / "DataSet/Data_All3",
    )
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=PROJECT_ROOT / "DataSet/Data_All/full_data_436-20260723.csv",
    )
    parser.add_argument(
        "--raw-feature-dir",
        type=Path,
        default=PROJECT_ROOT / "DataSet/Data_All/2_raw_features",
    )
    parser.add_argument("--train-rea", default="REA-1")
    parser.add_argument("--test-rea", nargs="+", default=["REA-2", "REA-3", "REA-4"])
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--target", choices=("ddg", "ee"), default="ddg")
    parser.add_argument("--run-name", default="rea1_train_rea234_test")
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def save_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle)


def select_features_with_target(
    feature: str,
    data_x: np.ndarray,
    labels: list[str],
    train_idx: np.ndarray,
    train_y: np.ndarray,
) -> tuple[list[int], list[str], dict[str, Any]]:
    train_x = data_x[train_idx]
    finite_mask = np.all(np.isfinite(train_x), axis=0)
    external_finite_mask = np.all(np.isfinite(data_x), axis=0)
    with np.errstate(over="ignore", invalid="ignore"):
        range_mask = np.all(np.abs(train_x) <= MAX_ABS, axis=0)
    finite_keep = finite_mask & external_finite_mask & range_mask
    finite_idx = np.flatnonzero(finite_keep).tolist()
    finite_labels = [labels[idx] for idx in finite_idx]
    dropped_nonfinite = [label for label, keep in zip(labels, finite_keep) if not keep]
    dropped_external_nonfinite = [
        label for label, train_finite, external_finite in zip(labels, finite_mask, external_finite_mask)
        if train_finite and not external_finite
    ]
    dropped_too_large = [
        label
        for label, finite, in_range in zip(labels, finite_mask, range_mask)
        if finite and not in_range
    ]
    finite_train_x = train_x[:, finite_idx]
    zero_keep: list[int] = []
    dropped_zero_std: list[str] = []
    for idx, label in enumerate(finite_labels):
        if label in ALWAYS_KEEP or not np.isclose(np.std(finite_train_x[:, idx]), 0.0, atol=1e-8):
            zero_keep.append(idx)
        else:
            dropped_zero_std.append(label)
    zero_labels = [finite_labels[idx] for idx in zero_keep]
    zero_train_x = finite_train_x[:, zero_keep]

    always_idx = [idx for idx, label in enumerate(zero_labels) if label in ALWAYS_KEEP]
    candidate_idx = [idx for idx, label in enumerate(zero_labels) if label not in ALWAYS_KEEP]
    config = FEATURE_CONFIG[feature]
    if feature == "xtb":
        selected_after_zero = list(range(len(zero_labels)))
        selector_info = {"method": "all", "k": None}
    else:
        k = config["k"]
        k_eff = min(int(k), len(candidate_idx)) if k is not None else len(candidate_idx)
        with np.errstate(divide="ignore", invalid="ignore"):
            scores, _ = f_regression(zero_train_x[:, candidate_idx], train_y)
        scores = np.asarray(scores, dtype=float)
        scores[~np.isfinite(scores)] = -np.inf
        order = np.argsort(scores)[::-1]
        selected_candidates = [candidate_idx[idx] for idx in order[:k_eff]]
        selected_after_zero = sorted(set(always_idx + selected_candidates))
        selector_info = {
            "method": "f_regression",
            "k": k,
            "effective_k": k_eff,
            "candidate_count": len(candidate_idx),
            "top_scores": {
                zero_labels[idx]: float(scores[pos])
                for pos, idx in zip(order[:k_eff], selected_candidates)
                if np.isfinite(scores[pos])
            },
        }

    selected_finite_idx = [zero_keep[idx] for idx in selected_after_zero]
    selected_data_idx = [finite_idx[idx] for idx in selected_finite_idx]
    selected_labels = [labels[idx] for idx in selected_data_idx]
    return selected_data_idx, selected_labels, {
        "finite_keep_indices": finite_idx,
        "finite_keep_labels": finite_labels,
        "dropped_nonfinite_features": dropped_nonfinite,
        "dropped_external_nonfinite_features": dropped_external_nonfinite,
        "dropped_too_large_features": dropped_too_large,
        "zero_std_keep_indices_after_finite": zero_keep,
        "zero_std_keep_labels": zero_labels,
        "dropped_zero_std_features": dropped_zero_std,
        "selected_indices_after_finite": selected_finite_idx,
        "selected_indices_original": selected_data_idx,
        "selected_features": selected_labels,
        "selector": selector_info,
    }


def load_component(
    feature: str,
    raw: pd.DataFrame,
    source: pd.DataFrame,
    train_rea: str,
    test_rea: list[str],
    target_col: str,
) -> dict[str, Any]:
    raw = raw.copy()
    raw["DATA_ID"] = raw["DATA_ID"].astype(str)
    source = source.copy()
    source["DATA_ID"] = source["DATA_ID"].astype(str)
    source_indexed = source.set_index("DATA_ID")
    raw_indexed = raw.set_index("DATA_ID")
    ids = source["DATA_ID"].tolist()
    if set(ids) != set(raw_indexed.index):
        raise ValueError(f"{feature} raw rows do not match the Data_All3 source subset")
    raw = raw_indexed.loc[ids].reset_index()
    batch_idx = raw.columns.get_loc("BATCH")
    class_idx = raw.columns.get_loc("CLASS")
    descriptor_labels = raw.columns[batch_idx + 1:class_idx].tolist()
    descriptor_labels = [
        label for label in descriptor_labels
        if label not in FEATURE_CONFIG[feature]["drop"]
    ]
    descriptor_x = raw.loc[:, descriptor_labels].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    condition_x = source_indexed.loc[ids, list(ALWAYS_KEEP)].apply(pd.to_numeric).to_numpy(float)
    data_x = np.concatenate([descriptor_x, condition_x], axis=1)
    labels = descriptor_labels + list(ALWAYS_KEEP)
    y = pd.to_numeric(source_indexed.loc[ids, target_col], errors="coerce").to_numpy(float)
    train_idx = np.flatnonzero(source_indexed.loc[ids, "REA_NAME"].to_numpy() == train_rea)
    rea_values = source_indexed.loc[ids, "REA_NAME"]
    test_idx = np.flatnonzero(rea_values.isin(test_rea).to_numpy())
    selected_idx, selected_labels, selector = select_features_with_target(
        feature, data_x, labels, train_idx, y[train_idx]
    )
    selected_train = data_x[train_idx][:, selected_idx]
    selected_test = data_x[test_idx][:, selected_idx]
    if not np.isfinite(selected_train).all() or not np.isfinite(selected_test).all():
        bad = [
            label for idx, label in enumerate(selected_labels)
            if not np.isfinite(np.concatenate([selected_train[:, idx], selected_test[:, idx]])).all()
        ]
        raise ValueError(f"Selected {feature} features contain non-finite test values: {bad[:10]}")
    return {
        "feature": feature,
        "train_x": selected_train,
        "test_x": selected_test,
        "train_y": y[train_idx],
        "test_y": y[test_idx],
        "train_idx": train_idx,
        "test_idx": test_idx,
        "selected_labels": selected_labels,
        "selector": selector,
        "data_class": pd.to_numeric(raw["CLASS"], errors="coerce").astype("Int64").tolist(),
    }


def main() -> None:
    args = parse_args()
    if args.train_rea in args.test_rea:
        raise ValueError("train_rea must not appear in test_rea")
    source = pd.read_csv(args.source_csv, low_memory=False)
    required = ["DATA_ID", "REA_NAME", "BATCH", "TEMP", "PRESSURE", "DDG", "EE"]
    missing = [name for name in required if name not in source.columns]
    if missing:
        raise ValueError(f"Source CSV is missing columns: {missing}")
    source["DATA_ID"] = source["DATA_ID"].astype(str)
    selected_rea = [args.train_rea] + list(args.test_rea)
    subset = source[source["REA_NAME"].isin(selected_rea) & (source["BATCH"] == args.batch)].copy()
    if subset["DATA_ID"].duplicated().any():
        raise ValueError("Source subset contains duplicated DATA_ID")
    expected_counts = {name: int((subset["REA_NAME"] == name).sum()) for name in selected_rea}
    if expected_counts[args.train_rea] == 0 or any(expected_counts[name] == 0 for name in args.test_rea):
        raise ValueError(f"Unexpected REA counts: {expected_counts}")
    target_col = "DDG" if args.target == "ddg" else "EE"
    if not np.isfinite(pd.to_numeric(subset[target_col], errors="coerce")).all():
        raise ValueError(f"Subset contains non-finite {target_col}")

    base = args.base
    raw_out = base / "2_raw_features"
    data_out = base / "3_data_for_train" / args.target / args.run_name / "rdkit_xtb_acsf"
    raw_out.mkdir(parents=True, exist_ok=True)
    data_out.mkdir(parents=True, exist_ok=True)
    subset_csv = base / "full_data_rea1_train_rea234_test-20260922.csv"
    subset.to_csv(subset_csv, index=False)

    ids = subset["DATA_ID"].tolist()
    components = {}
    for feature, filename in RAW_FILES.items():
        raw = pd.read_csv(args.raw_feature_dir / filename, low_memory=False)
        raw["DATA_ID"] = raw["DATA_ID"].astype(str)
        filtered = raw[raw["DATA_ID"].isin(ids)].set_index("DATA_ID").loc[ids].reset_index()
        if len(filtered) != len(subset):
            raise ValueError(f"Raw feature row count mismatch for {feature}")
        filtered.to_csv(raw_out / filename, index=False)
        components[feature] = load_component(
            feature, filtered, subset, args.train_rea, list(args.test_rea), target_col
        )

    train_parts = [components["rdkit"]["train_x"]]
    test_parts = [components["rdkit"]["test_x"]]
    labels = {"label1_rdkit": components["rdkit"]["selected_labels"]}
    dropped_conditions = {}
    label_index = 2
    for feature in ("xtb", "acsf"):
        component = components[feature]
        keep = [idx for idx, label in enumerate(component["selected_labels"]) if label not in ALWAYS_KEEP]
        train_parts.append(component["train_x"][:, keep])
        test_parts.append(component["test_x"][:, keep])
        labels[f"label{label_index}_{feature}"] = [component["selected_labels"][idx] for idx in keep]
        dropped_conditions[feature] = [label for label in component["selected_labels"] if label in ALWAYS_KEEP]
        label_index += 1

    train_x = np.concatenate(train_parts, axis=1)
    test_x = np.concatenate(test_parts, axis=1)
    train_idx = components["rdkit"]["train_idx"]
    test_idx = components["rdkit"]["test_idx"]
    train_names = subset.iloc[train_idx]["DATA_ID"].astype(str).tolist()
    test_names = subset.iloc[test_idx]["DATA_ID"].astype(str).tolist()
    train_meta = subset.iloc[train_idx]
    test_meta = subset.iloc[test_idx]
    if len(train_names) != expected_counts[args.train_rea] or len(test_names) != sum(expected_counts[n] for n in args.test_rea):
        raise ValueError("Unexpected train/test row counts")
    if set(train_names) & set(test_names):
        raise ValueError("Train/test DATA_ID overlap")
    if not np.isfinite(train_x).all() or not np.isfinite(test_x).all():
        raise ValueError("Combined train/test matrix contains NaN or inf")

    np.save(data_out / "train_data_x.npy", train_x)
    np.save(data_out / "train_data_y.npy", train_meta[target_col].to_numpy(float))
    np.save(data_out / "test_data_x.npy", test_x)
    np.save(data_out / "test_data_y.npy", test_meta[target_col].to_numpy(float))
    for prefix, frame, names in (("train", train_meta, train_names), ("test", test_meta, test_names)):
        save_pickle(data_out / f"{prefix}_x_label.pkl", labels)
        save_pickle(data_out / f"{prefix}_data_name.pkl", names)
        save_pickle(data_out / f"{prefix}_data_batch.pkl", frame["BATCH"].astype(int).tolist())
        save_pickle(data_out / f"{prefix}_data_class.pkl", [int(components["rdkit"]["data_class"][idx]) for idx in (train_idx if prefix == "train" else test_idx)])

    selector = {
        "version": 1,
        "kind": "fixed_rea_external_split",
        "target": args.target,
        "target_column": target_col,
        "train_rea": args.train_rea,
        "test_rea": list(args.test_rea),
        "batch": args.batch,
        "source_csv": str(args.source_csv),
        "subset_csv": str(subset_csv),
        "raw_feature_source": str(args.raw_feature_dir),
        "selected_features": labels,
        "components": {feature: component["selector"] for feature, component in components.items()},
        "dropped_duplicate_conditions": dropped_conditions,
        "row_counts": {
            "source_subset": len(subset),
            "train": len(train_names),
            "test": len(test_names),
            "test_by_rea": {name: int((test_meta["REA_NAME"] == name).sum()) for name in args.test_rea},
        },
        "feature_counts": {
            "rdkit": len(labels["label1_rdkit"]),
            "xtb": len(labels["label2_xtb"]),
            "acsf": len(labels["label3_acsf"]),
            "combined": int(train_x.shape[1]),
        },
        "train_data_ids": train_names,
        "test_data_ids": test_names,
    }
    write_json(data_out / "selector.json", selector)
    write_json(data_out / "manifest.json", selector | {"output_dir": str(data_out)})
    print(json.dumps({"subset_csv": str(subset_csv), "output_dir": str(data_out), "row_counts": selector["row_counts"], "feature_counts": selector["feature_counts"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
