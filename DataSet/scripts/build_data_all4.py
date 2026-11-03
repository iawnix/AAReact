#!/usr/bin/env python3
"""Build the frozen-split Data_All4 dataset.

Data_All BATCH=0 train/test membership is kept exactly as recorded in the
primary manifest. Labeled BATCH=2 rows from the 2026-10-28 file are appended
to the training side, and labeled BATCH=3 rows become the extra test set.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_regression


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ALL = PROJECT_ROOT / "DataSet" / "Data_All"
DATA_ALL2 = PROJECT_ROOT / "DataSet" / "Data_All2"
NEW_DATA = PROJECT_ROOT / "DataSet" / "Data_Raw" / "2026-10-28" / "data.csv"
OUT = PROJECT_ROOT / "DataSet" / "Data_All4"
TARGET = "ddg"
TARGET_COLUMN = "DDG"
SPLIT_NAME = "seed_1_test_0-2"
PRIMARY_MANIFEST = DATA_ALL / "3_data_for_train" / TARGET / SPLIT_NAME / "rdkit" / "manifest.json"

RAW_FILES = {
    "rdkit": "rdkit_desc_features.csv",
    "xtb": "xtb_features.csv",
    "soap": "soap_features.csv",
    "acsf": "acsf_features.csv",
    "rdkit_morgan": "rdkit_morgan_features.csv",
}
FEATURE_CONFIG = {
    "rdkit": (256, ("CAT_Ipc",)),
    "xtb": (None, ()),
    "soap": (256, ()),
    "acsf": (128, ()),
}
COMBINATIONS = [
    ("rdkit", "soap"),
    ("soap", "xtb"),
    ("rdkit", "xtb"),
    ("soap", "acsf"),
    ("acsf", "xtb"),
    ("rdkit", "acsf"),
    ("rdkit", "soap", "xtb"),
    ("rdkit", "soap", "acsf"),
    ("rdkit", "xtb", "acsf"),
    ("soap", "xtb", "acsf"),
    ("rdkit", "soap", "xtb", "acsf"),
]
ALWAYS_KEEP = ("TEMP", "PRESSURE")
SOURCE_COLUMNS = [
    "DATA_ID", "CAT_NAME", "SOL_NAME", "PRO_R_NAME", "PRO_S_NAME", "REA_NAME",
    "CAT_SMI", "SOL_SMI", "PRO_R_SMI", "PRO_S_SMI", "REA_SMI", "TEMP",
    "PRESSURE", "EE", "DDG", "CONV", "BATCH",
]
MAX_ABS = float(np.finfo(np.float32).max)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, low_memory=False)
    if "DATA_ID" not in frame.columns:
        raise ValueError(f"{path} is missing DATA_ID")
    frame["DATA_ID"] = frame["DATA_ID"].astype(str)
    if frame["DATA_ID"].duplicated().any():
        raise ValueError(f"{path} contains duplicate DATA_ID values")
    return frame


def assert_same_columns(left: pd.DataFrame, right: pd.DataFrame, label: str) -> None:
    if left.columns.tolist() != right.columns.tolist():
        raise ValueError(f"{label} column schema differs")


def assert_equal_values(left: pd.Series, right: pd.Series, label: str) -> None:
    l = left.astype(object).where(left.notna(), "<NA>").astype(str)
    r = right.astype(object).where(right.notna(), "<NA>").astype(str)
    mismatch = l != r
    if mismatch.any():
        ids = left.index[mismatch].astype(str).tolist()[:10]
        raise ValueError(f"{label} differs for DATA_ID values: {ids}")


def build_master() -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], list[str]]:
    all_frame = read_csv(DATA_ALL / "full_data_436-20260723.csv")
    all2_frame = read_csv(DATA_ALL2 / "full_data_918-20260723.csv")
    new_frame = read_csv(NEW_DATA)
    assert_same_columns(all_frame, all2_frame, "Data_All/Data_All2 source")
    assert_same_columns(all_frame, new_frame, "source/new data")

    overlap = set(all_frame["DATA_ID"]) & set(all2_frame["DATA_ID"])
    if overlap:
        raise ValueError(f"Data_All and Data_All2 overlap in DATA_ID: {sorted(overlap)[:10]}")

    master = pd.concat([all_frame, all2_frame], ignore_index=True)
    master = master.set_index("DATA_ID", drop=False)
    new_indexed = new_frame.set_index("DATA_ID", drop=False)
    common = master.index.intersection(new_indexed.index)
    if len(common) != len(new_frame):
        missing = sorted(set(new_indexed.index) - set(master.index))
        raise ValueError(f"New data contains DATA_ID values absent from Data_All2: {missing[:10]}")

    structural = [c for c in SOURCE_COLUMNS if c not in {"EE", "DDG", "CONV", "BATCH"}]
    for col in structural:
        assert_equal_values(master.loc[common, col], new_indexed.loc[common, col], f"new {col}")

    # New labels and the new batch role are authoritative for the 2026-10-28 rows.
    for col in ("EE", "DDG", "CONV", "BATCH"):
        master.loc[common, col] = new_indexed.loc[common, col].to_numpy()
    master = master.reset_index(drop=True)

    new2 = new_frame.loc[new_frame["BATCH"].eq(2), "DATA_ID"].astype(str).tolist()
    new3 = new_frame.loc[new_frame["BATCH"].eq(3), "DATA_ID"].astype(str).tolist()
    if set(new2) | set(new3) != set(common):
        raise ValueError("New data must contain exactly the rows whose labels are being overlaid")
    for col in ("EE", "DDG"):
        values = pd.to_numeric(new_frame[col], errors="coerce")
        if not np.isfinite(values).all():
            raise ValueError(f"New labeled data contains non-finite {col}")
    return master, new_frame, new2, new3, sorted(common.tolist())


def copy_sdf_files() -> None:
    out_sdf = OUT / "1_sdf"
    out_sdf.mkdir(parents=True, exist_ok=True)
    sources = [DATA_ALL / "1_sdf", DATA_ALL2 / "1_sdf"]
    copied: dict[str, str] = {}
    for source in sources:
        for path in sorted(source.glob("*.sdf")):
            destination = out_sdf / path.name
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if destination.exists():
                existing = hashlib.sha256(destination.read_bytes()).hexdigest()
                if existing != digest:
                    raise ValueError(f"Conflicting SDF files for {path.name}")
            else:
                destination.write_bytes(path.read_bytes())
            copied[path.name] = str(path)
    (OUT / "sdf_sources.json").write_text(json.dumps(copied, indent=2, ensure_ascii=False) + "\n")


def merge_raw_features(master: pd.DataFrame) -> None:
    out_raw = OUT / "2_raw_features"
    out_raw.mkdir(parents=True, exist_ok=True)
    master_indexed = master.set_index("DATA_ID", drop=False)
    for feature_name, filename in RAW_FILES.items():
        left = read_csv(DATA_ALL / "2_raw_features" / filename)
        right = read_csv(DATA_ALL2 / "2_raw_features" / filename)
        if left.columns.tolist() != right.columns.tolist():
            raise ValueError(f"{feature_name} raw feature schema differs between Data_All and Data_All2")
        if set(left.DATA_ID) & set(right.DATA_ID):
            raise ValueError(f"{feature_name} raw features overlap in DATA_ID")
        raw = pd.concat([left, right], ignore_index=True)
        raw = raw.set_index("DATA_ID", drop=False)
        if set(raw.index) != set(master_indexed.index):
            raise ValueError(f"{feature_name} raw feature IDs do not match merged source IDs")
        raw = raw.loc[master_indexed.index].copy()
        for col in SOURCE_COLUMNS:
            raw[col] = master_indexed[col].to_numpy()
        raw.reset_index(drop=True).to_csv(out_raw / filename, index=False)


def finite_filter(x: np.ndarray, labels: list[str]) -> tuple[list[int], list[str], list[str]]:
    finite = np.all(np.isfinite(x), axis=0)
    with np.errstate(over="ignore", invalid="ignore"):
        in_range = np.all(np.abs(x) <= MAX_ABS, axis=0)
    keep = finite & in_range
    kept = np.flatnonzero(keep).tolist()
    dropped_nonfinite = [label for label, ok in zip(labels, finite) if not ok]
    dropped_large = [label for label, ok, rng in zip(labels, finite, in_range) if ok and not rng]
    return kept, dropped_nonfinite, dropped_large


def select_component(
    feature: str,
    raw: pd.DataFrame,
    master: pd.DataFrame,
    train_ids: list[str],
    test_ids: list[str],
    extra_ids: list[str],
) -> dict[str, Any]:
    batch_idx = raw.columns.get_loc("BATCH")
    class_idx = raw.columns.get_loc("CLASS")
    raw_descriptor_labels = raw.columns[batch_idx + 1:class_idx].tolist()
    raw_feature_count = len(raw_descriptor_labels) + len(ALWAYS_KEEP)
    labels = raw_descriptor_labels
    k, drop = FEATURE_CONFIG[feature]
    labels = [label for label in labels if label not in drop]
    descriptor_x = raw.loc[:, labels].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    metadata = master.set_index("DATA_ID").loc[raw["DATA_ID"].tolist()]
    data_x = np.concatenate([
        descriptor_x,
        metadata.loc[:, list(ALWAYS_KEEP)].apply(pd.to_numeric, errors="coerce").to_numpy(float),
    ], axis=1)
    labels = labels + list(ALWAYS_KEEP)
    names = raw["DATA_ID"].astype(str).tolist()
    index = {name: idx for idx, name in enumerate(names)}
    train_idx = np.asarray([index[name] for name in train_ids], dtype=int)
    test_idx = np.asarray([index[name] for name in test_ids], dtype=int)
    extra_idx = np.asarray([index[name] for name in extra_ids], dtype=int)
    y = pd.to_numeric(metadata[TARGET_COLUMN], errors="coerce").to_numpy(float)
    train_y = y[train_idx]

    finite_idx, dropped_nonfinite, dropped_large = finite_filter(data_x[train_idx], labels)
    finite_labels = [labels[idx] for idx in finite_idx]
    train_finite = data_x[train_idx][:, finite_idx]
    test_finite = data_x[test_idx][:, finite_idx]
    extra_finite = data_x[extra_idx][:, finite_idx]

    zero_keep = []
    dropped_zero = []
    for idx, label in enumerate(finite_labels):
        if label in ALWAYS_KEEP or not np.isclose(np.std(train_finite[:, idx]), 0.0, atol=1e-8):
            zero_keep.append(idx)
        else:
            dropped_zero.append(label)
    zero_labels = [finite_labels[idx] for idx in zero_keep]
    train_zero = train_finite[:, zero_keep]

    always_idx = [idx for idx, label in enumerate(zero_labels) if label in ALWAYS_KEEP]
    candidate_idx = [idx for idx, label in enumerate(zero_labels) if label not in ALWAYS_KEEP]
    if feature == "xtb":
        selected_zero = list(range(len(zero_labels)))
        selector_info: dict[str, Any] = {"method": "all", "k": None}
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            scores, _ = f_regression(train_zero[:, candidate_idx], train_y)
        scores = np.asarray(scores, dtype=float)
        scores[~np.isfinite(scores)] = -np.inf
        order = np.argsort(scores)[::-1]
        k_eff = min(int(k), len(candidate_idx)) if k is not None else len(candidate_idx)
        selected_candidates = [candidate_idx[pos] for pos in order[:k_eff]]
        selected_zero = sorted(set(always_idx + selected_candidates))
        selector_info = {
            "method": "f_regression",
            "k": k,
            "effective_k": k_eff,
            "candidate_count": len(candidate_idx),
            "top_scores": {
                zero_labels[candidate_idx[pos]]: float(scores[pos])
                for pos in order[:k_eff] if np.isfinite(scores[pos])
            },
        }

    selected_finite = [zero_keep[idx] for idx in selected_zero]
    selected_data = [finite_idx[idx] for idx in selected_finite]
    selected_labels = [labels[idx] for idx in selected_data]
    train_x = data_x[train_idx][:, selected_data]
    test_x = data_x[test_idx][:, selected_data]
    extra_x = data_x[extra_idx][:, selected_data]
    for name, matrix in (("train", train_x), ("test", test_x), ("extra", extra_x)):
        if not np.isfinite(matrix).all():
            raise ValueError(f"Selected {feature} features contain non-finite {name} values")

    return {
        "feature": feature,
        "raw_feature_count": raw_feature_count,
        "train_x": train_x,
        "test_x": test_x,
        "extra_x": extra_x,
        "train_y": y[train_idx],
        "test_y": y[test_idx],
        "extra_y": y[extra_idx],
        "train_batch": metadata.loc[train_ids, "BATCH"].astype(int).tolist(),
        "test_batch": metadata.loc[test_ids, "BATCH"].astype(int).tolist(),
        "extra_batch": metadata.loc[extra_ids, "BATCH"].astype(int).tolist(),
        "labels": selected_labels,
        "class": pd.to_numeric(raw.set_index("DATA_ID").loc[train_ids + test_ids + extra_ids, "CLASS"], errors="coerce").astype("Int64").tolist(),
        "selector": {
            "finite_keep_indices": finite_idx,
            "finite_keep_labels": finite_labels,
            "dropped_nonfinite_features": dropped_nonfinite,
            "dropped_too_large_features": dropped_large,
            "zero_std_keep_labels": zero_labels,
            "dropped_zero_std_features": dropped_zero,
            "selected_features": selected_labels,
            "target_selector": selector_info,
        },
    }


def save_pickle(path: Path, value: Any) -> None:
    with path.open("wb") as handle:
        pickle.dump(value, handle)


def save_part(root: Path, prefix: str, x: np.ndarray, y: np.ndarray, labels: dict[str, list[str]], names: list[str], batches: list[int], classes: list[Any]) -> None:
    np.save(root / f"{prefix}_data_x.npy", x)
    np.save(root / f"{prefix}_data_y.npy", y)
    save_pickle(root / f"{prefix}_x_label.pkl", labels)
    save_pickle(root / f"{prefix}_data_name.pkl", names)
    save_pickle(root / f"{prefix}_data_batch.pkl", batches)
    save_pickle(root / f"{prefix}_data_class.pkl", classes)


def save_artifacts(root: Path, manifest: dict[str, Any], selector: dict[str, Any]) -> None:
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    save_pickle(root / "selector.pkl", selector)
    (root / "selector.json").write_text(json.dumps(selector, indent=2, ensure_ascii=False) + "\n")


def save_component(
    root: Path,
    component: dict[str, Any],
    labels: dict[str, list[str]],
    train_ids: list[str],
    test_ids: list[str],
    extra_ids: list[str],
    artifacts: tuple[dict[str, Any], dict[str, Any]] | None = None,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    train_test_ids = train_ids + test_ids
    train_test_x = np.concatenate([component["train_x"], component["test_x"]], axis=0)
    train_test_y = np.concatenate([component["train_y"], component["test_y"]], axis=0)
    train_test_batches = component["train_batch"] + component["test_batch"]
    save_part(root, "train", component["train_x"], component["train_y"], labels, train_ids, component["train_batch"], component["class"][:len(train_ids)])
    save_part(root, "test", component["test_x"], component["test_y"], labels, test_ids, component["test_batch"], component["class"][len(train_ids):len(train_ids)+len(test_ids)])
    save_part(root, "extra", component["extra_x"], component["extra_y"], labels, extra_ids, component["extra_batch"], component["class"][len(train_test_ids):])
    save_part(root, "train_test", train_test_x, train_test_y, labels, train_test_ids, train_test_batches, component["class"][:len(train_test_ids)])
    if artifacts is not None:
        save_artifacts(root, *artifacts)


def single_artifacts(feature: str, component: dict[str, Any], labels: dict[str, list[str]], train_ids: list[str], test_ids: list[str], extra_ids: list[str], raw_rows: int) -> tuple[dict[str, Any], dict[str, Any]]:
    created_at = datetime.now().isoformat(timespec="seconds")
    selector = dict(component["selector"])
    selector.update({
        "finite_filter": "drop features with non-finite values or abs(value) > max_abs_feature_value in train rows only",
        "zero_std_atol": 1e-8,
        "max_abs_feature_value": MAX_ABS,
        "dropped_manual_features": list(FEATURE_CONFIG[feature][1]),
    })
    manifest = {
        "version": 1,
        "created_at": created_at,
        "target": TARGET,
        "target_column": TARGET_COLUMN,
        "feature": feature,
        "source_feature_csv": str(OUT / "2_raw_features" / RAW_FILES[feature]),
        "source_target_csv": str(OUT / "full_data_merged_20261102.csv"),
        "seed": 1,
        "test_size": 0.2,
        "train_batches": [0, 2],
        "extra_batches": [3],
        "always_keep_features": list(ALWAYS_KEEP),
        "row_counts": {"raw": raw_rows, "target_na": 830, "train_test": len(train_ids) + len(test_ids), "train": len(train_ids), "test": len(test_ids), "extra": len(extra_ids)},
        "feature_counts": {
            "raw": component["raw_feature_count"],
            "finite": len(component["selector"]["finite_keep_labels"]),
            "zero_std_kept": len(component["selector"]["zero_std_keep_labels"]),
            "selected": len(component["labels"]),
        },
        "selector": selector,
        "selected_features": component["labels"],
        "train_ids": train_ids,
        "test_ids": test_ids,
        "extra_ids": extra_ids,
    }
    selector_artifact = {
        "version": 1,
        "kind": "single_feature_selector",
        "created_at": created_at,
        "target": TARGET,
        "target_column": TARGET_COLUMN,
        "feature": feature,
        "source_feature_csv": manifest["source_feature_csv"],
        "source_target_csv": manifest["source_target_csv"],
        "seed": 1,
        "test_size": 0.2,
        "train_batches": [0, 2],
        "extra_batches": [3],
        "selected_label_payload": labels,
        "selected_features": component["labels"],
        "source_schema": component["selector"]["finite_keep_labels"],
        "finite_keep_indices": component["selector"]["finite_keep_indices"],
        "finite_keep_labels": component["selector"]["finite_keep_labels"],
        "zero_std_keep_labels": component["selector"]["zero_std_keep_labels"],
        "selector": selector,
        "train_ids": train_ids,
        "test_ids": test_ids,
        "extra_ids": extra_ids,
    }
    return manifest, selector_artifact


def combine_components(root: Path, components: dict[str, dict[str, Any]], combo: tuple[str, ...], train_ids: list[str], test_ids: list[str], extra_ids: list[str]) -> None:
    parts = []
    labels: dict[str, list[str]] = {}
    for idx, feature in enumerate(combo, start=1):
        item = components[feature]
        keep = [i for i, label in enumerate(item["labels"]) if idx == 1 or label not in ALWAYS_KEEP]
        parts.append({
            "train_x": item["train_x"][:, keep],
            "test_x": item["test_x"][:, keep],
            "extra_x": item["extra_x"][:, keep],
        })
        labels[f"label{idx}_{feature}"] = [item["labels"][i] for i in keep]
    combined = {
        "train_x": np.concatenate([item["train_x"] for item in parts], axis=1),
        "test_x": np.concatenate([item["test_x"] for item in parts], axis=1),
        "extra_x": np.concatenate([item["extra_x"] for item in parts], axis=1),
        "train_y": components[combo[0]]["train_y"],
        "test_y": components[combo[0]]["test_y"],
        "extra_y": components[combo[0]]["extra_y"],
        "train_batch": components[combo[0]]["train_batch"],
        "test_batch": components[combo[0]]["test_batch"],
        "extra_batch": components[combo[0]]["extra_batch"],
        "class": components[combo[0]]["class"],
        "labels": labels,
    }
    component_manifests = [str(root.parent / feature / "manifest.json") for feature in combo]
    component_selectors = [str(root.parent / feature / "selector.json") for feature in combo]
    created_at = datetime.now().isoformat(timespec="seconds")
    manifest = {
        "version": 1,
        "created_at": created_at,
        "target": TARGET,
        "target_column": TARGET_COLUMN,
        "seed": 1,
        "test_size": 0.2,
        "combination": list(combo),
        "source_target_csv": str(OUT / "full_data_merged_20261102.csv"),
        "component_manifests": component_manifests,
        "component_selectors": component_selectors,
        "row_counts": {"train_test": len(train_ids) + len(test_ids), "train": len(train_ids), "test": len(test_ids), "extra": len(extra_ids)},
        "feature_counts": {"train_test": int(combined["train_x"].shape[1]), "extra": int(combined["extra_x"].shape[1])},
    }
    selector = {
        "version": 1,
        "kind": "combined_feature_selector",
        "created_at": created_at,
        "target": TARGET,
        "seed": 1,
        "test_size": 0.2,
        "combination": list(combo),
        "source_target_csv": manifest["source_target_csv"],
        "component_manifests": component_manifests,
        "component_selectors": component_selectors,
        "selected_label_payload": labels,
    }
    save_component(root / "_tmp", combined, labels, train_ids, test_ids, extra_ids, (manifest, selector))
    tmp = root / "_tmp"
    for path in tmp.iterdir():
        destination = root / path.name
        path.replace(destination)
    tmp.rmdir()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    master, new_frame, new2_ids, new3_ids, _ = build_master()
    primary = json.loads(PRIMARY_MANIFEST.read_text())
    old_train_ids = [str(item) for item in primary["train_ids"]]
    old_test_ids = [str(item) for item in primary["test_ids"]]
    if set(old_train_ids) & set(old_test_ids):
        raise ValueError("Original train/test IDs overlap")
    if set(new2_ids) & (set(old_train_ids) | set(old_test_ids)):
        raise ValueError("New BATCH=2 IDs overlap original Data_All IDs")
    if set(new3_ids) & (set(old_train_ids) | set(old_test_ids)):
        raise ValueError("New BATCH=3 IDs overlap original Data_All IDs")
    train_ids = old_train_ids + new2_ids
    test_ids = old_test_ids
    extra_ids = new3_ids

    master.to_csv(OUT / "full_data_merged_20261102.csv", index=False)
    copy_sdf_files()
    merge_raw_features(master)

    raw_dir = OUT / "2_raw_features"
    components: dict[str, dict[str, Any]] = {}
    for feature in FEATURE_CONFIG:
        raw = read_csv(raw_dir / RAW_FILES[feature])
        component = select_component(feature, raw, master, train_ids, test_ids, extra_ids)
        components[feature] = component
        labels = {f"label1_{feature}": component["labels"]}
        artifacts = single_artifacts(feature, component, labels, train_ids, test_ids, extra_ids, len(raw))
        save_component(OUT / "3_data_for_train" / TARGET / SPLIT_NAME / feature, component, labels, train_ids, test_ids, extra_ids, artifacts)

    dataset_root = OUT / "3_data_for_train" / TARGET / SPLIT_NAME
    for combo in COMBINATIONS:
        combo_name = "_".join(combo)
        combine_components(dataset_root / combo_name, components, combo, train_ids, test_ids, extra_ids)

    summary = {
        "version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "target": TARGET,
        "target_column": TARGET_COLUMN,
        "source_data_all": str(DATA_ALL / "full_data_436-20260723.csv"),
        "source_data_all2": str(DATA_ALL2 / "full_data_918-20260723.csv"),
        "source_new_labels": str(NEW_DATA),
        "merged_source": str(OUT / "full_data_merged_20261102.csv"),
        "primary_seed": 1,
        "primary_test_size": 0.2,
        "split_policy": "freeze original Data_All train/test IDs; append new BATCH=2 to train; BATCH=3 is extra",
        "row_counts": {
            "merged_source": int(len(master)),
            "original_train": len(old_train_ids),
            "new_batch_2_train": len(new2_ids),
            "train": len(train_ids),
            "test": len(test_ids),
            "extra": len(extra_ids),
            "unlabeled_data_all2": int(master["DDG"].isna().sum()),
        },
        "train_ids": train_ids,
        "test_ids": test_ids,
        "extra_ids": extra_ids,
        "new_batch_2_ids": new2_ids,
        "new_batch_3_ids": new3_ids,
        "descriptors": list(FEATURE_CONFIG) + ["rdkit_morgan (raw only)"],
    }
    (OUT / "manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: summary[k] for k in ("merged_source", "row_counts", "new_batch_2_ids", "new_batch_3_ids")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
