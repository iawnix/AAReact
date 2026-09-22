#!/usr/bin/env python3
import argparse
import json
import pickle
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

from config.constants import normalize_target, target_column
from util.feature_transform import nonfinite_labels


ALWAYS_KEEP_FEATURES = ["TEMP", "PRESSURE"]
DEFAULT_MAX_ABS_FEATURE_VALUE = float(np.finfo(np.float32).max)


@dataclass(frozen=True)
class FeatureConfig:
    file_name: str
    selector: str
    k: int | None
    drop_labels: tuple[str, ...] = ()


FEATURE_CONFIGS = {
    "rdkit": FeatureConfig("rdkit_desc_features.csv", "f_regression", 256, ("CAT_Ipc",)),
    "xtb": FeatureConfig("xtb_features.csv", "all", None),
    "soap": FeatureConfig("soap_features.csv", "f_regression", 256),
    "acsf": FeatureConfig("acsf_features.csv", "f_regression", 128),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build target-specific ML datasets from Data_All raw feature CSV files.")
    parser.add_argument("--base", type=Path, default=Path("/home/iaw/DATA2/AAReact/DataSet/Data_All"))
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=None,
        help="Authoritative source CSV for DATA_ID/TEMP/PRESSURE/BATCH/target values. Defaults to base/full_data_436-20260723.csv.",
    )
    parser.add_argument("--target", choices=["ee", "ddg"], required=True)
    parser.add_argument("--features", nargs="+", default=list(FEATURE_CONFIGS.keys()), choices=list(FEATURE_CONFIGS.keys()))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--train-batches", nargs="+", type=int, default=[0])
    parser.add_argument("--extra-batches", nargs="+", type=int, default=[1])
    parser.add_argument("--rdkit-k", type=int, default=FEATURE_CONFIGS["rdkit"].k)
    parser.add_argument("--soap-k", type=int, default=FEATURE_CONFIGS["soap"].k)
    parser.add_argument("--acsf-k", type=int, default=FEATURE_CONFIGS["acsf"].k)
    parser.add_argument("--zero-std-atol", type=float, default=1e-8)
    parser.add_argument(
        "--max-abs-feature-value",
        type=float,
        default=DEFAULT_MAX_ABS_FEATURE_VALUE,
        help="Drop train-fitted feature columns whose absolute value exceeds this threshold.",
    )
    return parser.parse_args()


def split_tag(seed: int, test_size: float) -> str:
    test_text = str(test_size).replace(".", "-")
    return "seed_{}_test_{}".format(seed, test_text)


def config_for(feature_name: str, args: argparse.Namespace) -> FeatureConfig:
    config = FEATURE_CONFIGS[feature_name]
    override_k = {
        "rdkit": args.rdkit_k,
        "soap": args.soap_k,
        "acsf": args.acsf_k,
    }.get(feature_name, config.k)
    return FeatureConfig(config.file_name, config.selector, override_k, config.drop_labels)


def load_source_metadata(source_csv: Path, target_col: str) -> pd.DataFrame:
    if not source_csv.exists():
        raise FileNotFoundError("Source CSV does not exist: {}".format(source_csv))
    source = pd.read_csv(source_csv, low_memory=False)
    required = ["DATA_ID", "TEMP", "PRESSURE", "BATCH", target_col]
    missing = [col for col in required if col not in source.columns]
    if missing:
        raise ValueError("{} is missing columns: {}".format(source_csv, ", ".join(missing)))
    if source["DATA_ID"].duplicated().any():
        duplicated = source["DATA_ID"][source["DATA_ID"].duplicated()].astype(str).tolist()
        raise ValueError("{} has duplicated DATA_ID values: {}".format(source_csv, duplicated[:10]))
    metadata = source[required].copy()
    metadata["DATA_ID"] = metadata["DATA_ID"].astype(str)
    return metadata


def load_feature_frame(
    feature_fp: Path,
    target_col: str,
    source_metadata: pd.DataFrame,
    drop_labels: tuple[str, ...] = (),
) -> dict[str, Any]:
    df = pd.read_csv(feature_fp, low_memory=False)
    required = ["DATA_ID", "BATCH", "CLASS"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError("{} is missing columns: {}".format(feature_fp, ", ".join(missing)))
    if df["DATA_ID"].duplicated().any():
        duplicated = df["DATA_ID"][df["DATA_ID"].duplicated()].tolist()
        raise ValueError("{} has duplicated DATA_ID values: {}".format(feature_fp, duplicated[:10]))
    df["DATA_ID"] = df["DATA_ID"].astype(str)

    batch_idx = df.columns.get_loc("BATCH")
    class_idx = df.columns.get_loc("CLASS")
    if class_idx <= batch_idx:
        raise ValueError("CLASS must appear after BATCH in {}".format(feature_fp))

    descriptor_labels = df.columns[batch_idx + 1:class_idx].to_list()
    manual_drop_labels = [label for label in drop_labels if label in descriptor_labels]
    descriptor_labels = [label for label in descriptor_labels if label not in manual_drop_labels]
    descriptor_x = df.loc[:, descriptor_labels].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    metadata = df[["DATA_ID"]].merge(source_metadata, on="DATA_ID", how="left", validate="one_to_one")
    missing_source = metadata.loc[metadata[target_col].isna(), "DATA_ID"].astype(str).tolist()
    if missing_source:
        raise ValueError("{} has DATA_ID values absent from source CSV metadata: {}".format(
            feature_fp, missing_source[:10]
        ))
    condition_x = metadata.loc[:, ALWAYS_KEEP_FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    data_x = np.concatenate([descriptor_x, condition_x], axis=1)
    x_label = descriptor_labels + ALWAYS_KEEP_FEATURES

    return {
        "data_x": data_x,
        "data_y": pd.to_numeric(metadata[target_col], errors="coerce").to_numpy(dtype=float),
        "x_label": x_label,
        "data_class": pd.to_numeric(df["CLASS"], errors="coerce").astype("Int64").tolist(),
        "data_name": df["DATA_ID"].astype(str).tolist(),
        "data_batch": pd.to_numeric(metadata["BATCH"], errors="raise").astype(int).tolist(),
        "manual_drop_labels": manual_drop_labels,
        "raw_rows": int(len(df)),
    }


def select_batch_indices(data_batch: list[int], data_y: np.ndarray, batches: list[int]) -> np.ndarray:
    batch_arr = np.asarray(data_batch)
    mask = np.isin(batch_arr, np.asarray(batches)) & np.isfinite(data_y)
    return np.flatnonzero(mask)


def finite_feature_filter(
    data_x: np.ndarray,
    x_label: list[str],
    max_abs_feature_value: float,
) -> tuple[np.ndarray, list[str], list[int], list[str], list[str]]:
    finite_mask = np.all(np.isfinite(data_x), axis=0)
    with np.errstate(over="ignore", invalid="ignore"):
        range_mask = np.all(np.abs(data_x) <= max_abs_feature_value, axis=0)
    keep_mask = finite_mask & range_mask
    keep_idx = np.flatnonzero(keep_mask).tolist()
    dropped_nonfinite = [label for label, keep in zip(x_label, finite_mask) if not keep]
    dropped_too_large = [
        label
        for label, finite, in_range in zip(x_label, finite_mask, range_mask)
        if finite and not in_range
    ]
    return data_x[:, keep_idx], [x_label[i] for i in keep_idx], keep_idx, dropped_nonfinite, dropped_too_large


def too_large_labels(data_x: np.ndarray, x_label: list[str], max_abs_feature_value: float) -> list[str]:
    if data_x.size == 0:
        return []
    with np.errstate(over="ignore", invalid="ignore"):
        too_large_mask = np.any(np.abs(data_x) > max_abs_feature_value, axis=0)
    return [label for label, too_large in zip(x_label, too_large_mask) if too_large]


def zero_std_filter(data_x_train: np.ndarray, x_label: list[str], atol: float) -> tuple[list[int], list[str]]:
    std = np.std(data_x_train, axis=0)
    keep_idx = []
    dropped = []
    for idx, label in enumerate(x_label):
        if label in ALWAYS_KEEP_FEATURES or not np.isclose(std[idx], 0.0, atol=atol):
            keep_idx.append(idx)
        else:
            dropped.append(label)
    return keep_idx, dropped


def target_filter(
    data_x_train: np.ndarray,
    data_y_train: np.ndarray,
    x_label: list[str],
    selector: str,
    k: int | None,
) -> tuple[list[int], dict[str, Any]]:
    if selector == "all":
        return list(range(len(x_label))), {"method": "all", "k": None}
    if selector != "f_regression":
        raise ValueError("Unsupported selector: {}".format(selector))

    always_idx = [idx for idx, label in enumerate(x_label) if label in ALWAYS_KEEP_FEATURES]
    candidate_idx = [idx for idx, label in enumerate(x_label) if label not in ALWAYS_KEEP_FEATURES]
    if not candidate_idx:
        return always_idx, {"method": selector, "k": k, "candidate_count": 0}

    k_eff = len(candidate_idx) if k is None else min(int(k), len(candidate_idx))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores, pvalues = f_regression(data_x_train[:, candidate_idx], data_y_train)
    scores = np.asarray(scores, dtype=float)
    scores[~np.isfinite(scores)] = -np.inf
    order = np.argsort(scores)[::-1]
    selected_candidate_idx = [candidate_idx[i] for i in order[:k_eff]]
    selected_idx = sorted(set(always_idx + selected_candidate_idx))

    selected_scores = {
        x_label[candidate_idx[i]]: float(scores[i])
        for i in order[:k_eff]
        if np.isfinite(scores[i])
    }
    return selected_idx, {
        "method": selector,
        "k": k,
        "effective_k": k_eff,
        "candidate_count": len(candidate_idx),
        "top_scores": selected_scores,
    }


def save_dataset(prefix: str, out_dir: Path, data_x: np.ndarray, data_y: np.ndarray, x_label: dict[str, list[str]],
                 data_class: list[Any], data_name: list[str], data_batch: list[int]) -> None:
    np.save(out_dir / "{}_data_x.npy".format(prefix), data_x)
    np.save(out_dir / "{}_data_y.npy".format(prefix), data_y)
    with open(out_dir / "{}_x_label.pkl".format(prefix), "wb") as f:
        pickle.dump(x_label, f)
    with open(out_dir / "{}_data_class.pkl".format(prefix), "wb") as f:
        pickle.dump(data_class, f)
    with open(out_dir / "{}_data_name.pkl".format(prefix), "wb") as f:
        pickle.dump(data_name, f)
    with open(out_dir / "{}_data_batch.pkl".format(prefix), "wb") as f:
        pickle.dump(data_batch, f)


def save_selector_artifact(out_dir: Path, artifact: dict[str, Any]) -> None:
    (out_dir / "selector.json").write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n")
    with open(out_dir / "selector.pkl", "wb") as f:
        pickle.dump(artifact, f)


def subset_list(values: list[Any], indices: np.ndarray) -> list[Any]:
    return [values[int(i)] for i in indices]


def build_one(feature_name: str, args: argparse.Namespace, source_metadata: pd.DataFrame, source_csv: Path) -> dict[str, Any]:
    target = normalize_target(args.target)
    target_col = target_column(target)
    config = config_for(feature_name, args)
    feature_fp = args.base / "2_raw_features" / config.file_name
    loaded = load_feature_frame(feature_fp, target_col, source_metadata, config.drop_labels)

    data_x = loaded["data_x"]
    data_y = loaded["data_y"]
    x_label = loaded["x_label"]
    data_class = loaded["data_class"]
    data_name = loaded["data_name"]
    data_batch = loaded["data_batch"]

    train_test_idx = select_batch_indices(data_batch, data_y, args.train_batches)
    extra_idx = select_batch_indices(data_batch, data_y, args.extra_batches)
    target_na_ids = [
        name for name, y in zip(data_name, data_y)
        if not np.isfinite(y)
    ]
    if len(train_test_idx) == 0:
        raise ValueError("No train/test rows selected for {}".format(feature_name))

    train_rel, test_rel = train_test_split(
        np.arange(len(train_test_idx)),
        test_size=args.test_size,
        random_state=args.seed,
    )

    train_test_x_raw = data_x[train_test_idx]
    extra_x_raw = data_x[extra_idx] if len(extra_idx) else np.empty((0, data_x.shape[1]))
    train_x_raw = train_test_x_raw[train_rel]
    _, finite_labels, finite_idx, dropped_nonfinite, dropped_too_large = finite_feature_filter(
        train_x_raw,
        x_label,
        args.max_abs_feature_value,
    )
    train_test_x_finite = train_test_x_raw[:, finite_idx]
    extra_x_finite = extra_x_raw[:, finite_idx] if len(extra_idx) else extra_x_raw[:, finite_idx]
    train_x_finite = train_test_x_finite[train_rel]
    train_y = data_y[train_test_idx][train_rel]

    zero_keep_idx, dropped_zero_std = zero_std_filter(train_x_finite, finite_labels, args.zero_std_atol)
    labels_after_zero = [finite_labels[i] for i in zero_keep_idx]
    train_x_zero = train_x_finite[:, zero_keep_idx]

    selected_after_zero_idx, target_selector_info = target_filter(
        train_x_zero,
        train_y,
        labels_after_zero,
        config.selector,
        config.k,
    )
    selected_finite_idx = [zero_keep_idx[i] for i in selected_after_zero_idx]
    selected_labels = [finite_labels[i] for i in selected_finite_idx]

    train_test_x = train_test_x_finite[:, selected_finite_idx]
    extra_x = extra_x_finite[:, selected_finite_idx] if len(extra_idx) else extra_x_finite
    bad_train_test_labels = nonfinite_labels(train_test_x, selected_labels)
    bad_extra_labels = nonfinite_labels(extra_x, selected_labels)
    bad_train_test_large_labels = too_large_labels(train_test_x, selected_labels, args.max_abs_feature_value)
    bad_extra_large_labels = too_large_labels(extra_x, selected_labels, args.max_abs_feature_value)
    if bad_train_test_labels:
        raise ValueError("Selected {} features contain non-finite train/test values: {}".format(
            feature_name, bad_train_test_labels[:10]
        ))
    if bad_extra_labels:
        raise ValueError("Selected {} features contain non-finite extra values: {}".format(
            feature_name, bad_extra_labels[:10]
        ))
    if bad_train_test_large_labels:
        raise ValueError("Selected {} features exceed max abs {} in train/test values: {}".format(
            feature_name, args.max_abs_feature_value, bad_train_test_large_labels[:10]
        ))
    if bad_extra_large_labels:
        raise ValueError("Selected {} features exceed max abs {} in extra values: {}".format(
            feature_name, args.max_abs_feature_value, bad_extra_large_labels[:10]
        ))

    out_dir = args.base / "3_data_for_train" / target / split_tag(args.seed, args.test_size) / feature_name
    out_dir.mkdir(parents=True, exist_ok=True)
    label_payload = {"label1_{}".format(feature_name): selected_labels}
    save_dataset(
        "train_test",
        out_dir,
        train_test_x,
        data_y[train_test_idx],
        label_payload,
        subset_list(data_class, train_test_idx),
        subset_list(data_name, train_test_idx),
        subset_list(data_batch, train_test_idx),
    )
    save_dataset(
        "extra",
        out_dir,
        extra_x,
        data_y[extra_idx],
        label_payload,
        subset_list(data_class, extra_idx),
        subset_list(data_name, extra_idx),
        subset_list(data_batch, extra_idx),
    )

    manifest = {
        "version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "target": target,
        "target_column": target_col,
        "feature": feature_name,
        "source_feature_csv": str(feature_fp),
        "source_target_csv": str(source_csv),
        "seed": args.seed,
        "test_size": args.test_size,
        "train_batches": args.train_batches,
        "extra_batches": args.extra_batches,
        "always_keep_features": ALWAYS_KEEP_FEATURES,
        "row_counts": {
            "raw": loaded["raw_rows"],
            "target_na": len(target_na_ids),
            "train_test": int(len(train_test_idx)),
            "train": int(len(train_rel)),
            "test": int(len(test_rel)),
            "extra": int(len(extra_idx)),
        },
        "feature_counts": {
            "raw": len(x_label),
            "finite": len(finite_labels),
            "zero_std_kept": len(labels_after_zero),
            "selected": len(selected_labels),
        },
        "selector": {
            "finite_filter": "drop features with non-finite values or abs(value) > max_abs_feature_value in train rows only",
            "zero_std_atol": args.zero_std_atol,
            "max_abs_feature_value": args.max_abs_feature_value,
            "dropped_nonfinite_features": dropped_nonfinite,
            "dropped_too_large_features": dropped_too_large,
            "dropped_manual_features": loaded["manual_drop_labels"],
            "dropped_zero_std_features": dropped_zero_std,
            "target_filter": target_selector_info,
        },
        "selected_features": selected_labels,
        "dropped_target_na_ids": target_na_ids,
        "train_ids": subset_list(data_name, train_test_idx[train_rel]),
        "test_ids": subset_list(data_name, train_test_idx[test_rel]),
        "extra_ids": subset_list(data_name, extra_idx),
    }
    selector_artifact = {
        "version": 1,
        "kind": "single_feature_selector",
        "created_at": manifest["created_at"],
        "target": target,
        "target_column": target_col,
        "feature": feature_name,
        "source_feature_csv": str(feature_fp),
        "source_target_csv": str(source_csv),
        "seed": args.seed,
        "test_size": args.test_size,
        "train_batches": args.train_batches,
        "extra_batches": args.extra_batches,
        "selected_label_payload": label_payload,
        "selected_features": selected_labels,
        "source_schema": x_label,
        "finite_keep_indices": finite_idx,
        "finite_keep_labels": finite_labels,
        "zero_std_keep_indices_after_finite": zero_keep_idx,
        "zero_std_keep_labels": labels_after_zero,
        "selected_indices_after_finite": selected_finite_idx,
        "selector": manifest["selector"],
        "train_ids": manifest["train_ids"],
        "test_ids": manifest["test_ids"],
        "extra_ids": manifest["extra_ids"],
    }
    save_selector_artifact(out_dir, selector_artifact)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return {
        "feature": feature_name,
        "out_dir": str(out_dir),
        "row_counts": manifest["row_counts"],
        "feature_counts": manifest["feature_counts"],
    }


def main() -> None:
    args = parse_args()
    target = normalize_target(args.target)
    target_col = target_column(target)
    source_csv = args.source_csv or (args.base / "full_data_436-20260723.csv")
    source_metadata = load_source_metadata(source_csv, target_col)
    summaries = [build_one(feature_name, args, source_metadata, source_csv) for feature_name in args.features]
    print(json.dumps({
        "target": args.target,
        "split": split_tag(args.seed, args.test_size),
        "source_csv": str(source_csv),
        "outputs": summaries,
    }, indent=2))


if __name__ == "__main__":
    main()
