#!/usr/bin/env python3
import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LEAKAGE_LABELS = {"EE", "DDG", "CONV"}
ALWAYS_KEEP_FEATURES = {"TEMP", "PRESSURE"}
FEATURE_TYPES = {"rdkit", "soap", "xtb", "acsf"}
DDG_R_KCAL = 0.001987
CELSIUS_TO_KELVIN = 273.15
DEFAULT_DDG_ATOL = 1e-6
DEFAULT_MAX_ABS_FEATURE_VALUE = float(np.finfo(np.float32).max)
DEFAULT_DUPLICATE_Y_ATOL = 1e-8
DEFAULT_DUPLICATE_X_DECIMALS = 10
DEFAULT_DUPLICATE_WARNING_LIMIT = 20
SOURCE_REACTION_KEY_COLUMNS = [
    "CAT_NAME",
    "SOL_NAME",
    "PRO_R_NAME",
    "PRO_S_NAME",
    "REA_NAME",
    "TEMP",
    "PRESSURE",
]


def split_tag(seed: int, test_size: float) -> str:
    return "seed_{}_test_{}".format(seed, str(test_size).replace(".", "-"))


def load_pickle(fp: Path) -> Any:
    with open(fp, "rb") as f:
        return pickle.load(f)


def flatten_labels(label_payload: dict[str, list[str]]) -> list[str]:
    labels: list[str] = []
    for key in sorted(label_payload.keys(), key=lambda item: int(item.split("_", 1)[0].replace("label", ""))):
        labels.extend(label_payload[key])
    return labels


def load_json(fp: Path) -> dict[str, Any]:
    return json.loads(fp.read_text())


def dataset_feature_parts(dataset_name: str) -> list[str]:
    parts = dataset_name.split("_")
    if not parts or any(part not in FEATURE_TYPES for part in parts):
        raise ValueError("Unsupported dataset name: {}".format(dataset_name))
    return parts


def assert_unique(values: list[Any], label: str, dataset_dir: Path) -> None:
    if len(values) != len(set(values)):
        raise ValueError("{} has duplicated {}".format(dataset_dir, label))


def assert_same(reference: Any, current: Any, label: str, dataset_dir: Path) -> None:
    if isinstance(reference, np.ndarray):
        ok = np.array_equal(reference, current, equal_nan=True)
    else:
        ok = reference == current
    if not ok:
        raise ValueError("{} {} differs from reference".format(dataset_dir, label))


def drop_duplicate_conditions(x: np.ndarray, labels: list[str]) -> tuple[np.ndarray, list[str]]:
    keep_idx = [idx for idx, label in enumerate(labels) if label not in ALWAYS_KEEP_FEATURES]
    return x[:, keep_idx], [labels[idx] for idx in keep_idx]


def validate_numeric_values(
    dataset_dir: Path,
    prefix: str,
    data_x: np.ndarray,
    data_y: np.ndarray,
    labels: list[str],
    names: list[Any],
    max_abs_feature_value: float,
) -> None:
    if not np.isfinite(data_y).all():
        bad_rows = np.flatnonzero(~np.isfinite(data_y))[:10]
        bad = ["{} y={}".format(names[int(idx)], data_y[int(idx)]) for idx in bad_rows]
        raise ValueError("{} {} has non-finite target values: {}".format(dataset_dir, prefix, bad))

    if not np.isfinite(data_x).all():
        bad_row, bad_col = np.where(~np.isfinite(data_x))
        bad = [
            "{} {}={}".format(names[int(row)], labels[int(col)], data_x[int(row), int(col)])
            for row, col in zip(bad_row[:10], bad_col[:10])
        ]
        raise ValueError("{} {} has non-finite feature values: {}".format(dataset_dir, prefix, bad))

    if data_x.size == 0:
        return
    with np.errstate(over="ignore", invalid="ignore"):
        too_large = np.abs(data_x) > max_abs_feature_value
    if too_large.any():
        bad_row, bad_col = np.where(too_large)
        bad = [
            "{} {}={}".format(names[int(row)], labels[int(col)], data_x[int(row), int(col)])
            for row, col in zip(bad_row[:10], bad_col[:10])
        ]
        raise ValueError(
            "{} {} feature values exceed max abs {}: {}".format(
                dataset_dir, prefix, max_abs_feature_value, bad
            )
        )


def calc_ddg_from_ee(temp_c: np.ndarray, ee: np.ndarray) -> np.ndarray:
    temp_k = temp_c + CELSIUS_TO_KELVIN
    return DDG_R_KCAL * temp_k * np.log((1.0 + ee) / (1.0 - ee))


def target_value_column(target: str) -> str:
    if target == "ee":
        return "EE"
    if target == "ddg":
        return "DDG"
    raise ValueError("Unsupported target: {}".format(target))


def summarize_duplicate_group(
    names: list[Any],
    values: np.ndarray,
    limit: int = DEFAULT_DUPLICATE_WARNING_LIMIT,
) -> str:
    pairs = [
        "{}={:.12g}".format(str(name), float(value))
        for name, value in zip(names[:limit], values[:limit])
    ]
    if len(names) > limit:
        pairs.append("... +{} more".format(len(names) - limit))
    return "; ".join(pairs)


def validate_source_label_uniqueness(
    source_csv: Path,
    target: str,
    duplicate_y_atol: float,
    warning_limit: int,
) -> list[str]:
    if not source_csv.exists():
        raise FileNotFoundError("Source CSV does not exist: {}".format(source_csv))

    df = pd.read_csv(source_csv)
    target_col = target_value_column(target)
    required = ["DATA_ID", target_col, *SOURCE_REACTION_KEY_COLUMNS]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError("{} is missing columns required for duplicate label validation: {}".format(
            source_csv, ", ".join(missing)
        ))

    warnings: list[str] = []
    conflicts: list[str] = []
    consistent_duplicates: list[str] = []
    for key, group in df.groupby(SOURCE_REACTION_KEY_COLUMNS, dropna=False):
        if len(group) < 2:
            continue

        names = group["DATA_ID"].astype(str).tolist()
        values = pd.to_numeric(group[target_col], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values)
        if not finite.all():
            bad = [names[int(idx)] for idx in np.flatnonzero(~finite)[:warning_limit]]
            raise ValueError("{} duplicate-label validation found non-finite {} values: {}".format(
                source_csv, target_col, bad
            ))

        y_range = float(np.max(values) - np.min(values))
        key_text = ", ".join("{}={}".format(col, value) for col, value in zip(SOURCE_REACTION_KEY_COLUMNS, key))
        payload = "{} | n={} | {}_range={:.12g} | {}".format(
            key_text,
            len(group),
            target_col,
            y_range,
            summarize_duplicate_group(names, values, warning_limit),
        )
        if y_range > duplicate_y_atol:
            conflicts.append(payload)
        else:
            consistent_duplicates.append(payload)

    if conflicts:
        examples = " || ".join(conflicts[:warning_limit])
        raise ValueError(
            "{} has source label conflicts for target {}: identical reaction keys map to different {} "
            "values beyond atol={}. conflicts={}, examples: {}".format(
                source_csv,
                target,
                target_col,
                duplicate_y_atol,
                len(conflicts),
                examples,
            )
        )

    if consistent_duplicates:
        warnings.append(
            "{} has {} duplicated reaction-key groups with consistent {} values within atol={}. "
            "examples: {}".format(
                source_csv,
                len(consistent_duplicates),
                target_col,
                duplicate_y_atol,
                " || ".join(consistent_duplicates[:warning_limit]),
            )
        )
    return warnings


def load_source_reaction_key_map(source_csv: Path) -> dict[str, tuple[Any, ...]]:
    if not source_csv.exists():
        raise FileNotFoundError("Source CSV does not exist: {}".format(source_csv))

    df = pd.read_csv(source_csv)
    required = ["DATA_ID", *SOURCE_REACTION_KEY_COLUMNS]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError("{} is missing columns required for source reaction-key mapping: {}".format(
            source_csv, ", ".join(missing)
        ))
    if df["DATA_ID"].duplicated().any():
        duplicated = df.loc[df["DATA_ID"].duplicated(), "DATA_ID"].astype(str).tolist()
        raise ValueError("{} has duplicated DATA_ID values: {}".format(source_csv, duplicated[:10]))

    out: dict[str, tuple[Any, ...]] = {}
    for _, row in df.iterrows():
        out[str(row["DATA_ID"])] = tuple(row[col] for col in SOURCE_REACTION_KEY_COLUMNS)
    return out


def load_and_validate_ddg_source(source_csv: Path, atol: float) -> dict[str, float]:
    if not source_csv.exists():
        raise FileNotFoundError("DDG source CSV does not exist: {}".format(source_csv))

    df = pd.read_csv(source_csv)
    required = ["DATA_ID", "TEMP", "EE", "DDG"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError("{} is missing columns required for ddG validation: {}".format(
            source_csv, ", ".join(missing)
        ))
    if df["DATA_ID"].duplicated().any():
        duplicated = df.loc[df["DATA_ID"].duplicated(), "DATA_ID"].astype(str).tolist()
        raise ValueError("{} has duplicated DATA_ID values: {}".format(source_csv, duplicated[:10]))

    data_id = df["DATA_ID"].astype(str).to_numpy()
    temp_c = pd.to_numeric(df["TEMP"], errors="coerce").to_numpy(dtype=float)
    ee = pd.to_numeric(df["EE"], errors="coerce").to_numpy(dtype=float)
    ddg = pd.to_numeric(df["DDG"], errors="coerce").to_numpy(dtype=float)

    finite_mask = np.isfinite(temp_c) & np.isfinite(ee) & np.isfinite(ddg)
    if not finite_mask.all():
        bad_ids = data_id[np.flatnonzero(~finite_mask)[:10]].tolist()
        raise ValueError("{} has non-finite TEMP/EE/DDG values for DATA_ID: {}".format(source_csv, bad_ids))

    ee_domain_mask = np.abs(ee) < 1.0
    if not ee_domain_mask.all():
        bad_ids = data_id[np.flatnonzero(~ee_domain_mask)[:10]].tolist()
        raise ValueError("{} has EE outside (-1, 1), cannot compute ddG: {}".format(source_csv, bad_ids))

    ddg_calc = calc_ddg_from_ee(temp_c, ee)
    diff = np.abs(ddg - ddg_calc)
    bad_idx = np.flatnonzero(diff > atol)
    if len(bad_idx):
        worst_idx = bad_idx[np.argsort(diff[bad_idx])[::-1][:10]]
        worst = [
            "{} TEMP={} EE={} DDG={} DDG_calc={} abs_diff={}".format(
                data_id[idx],
                temp_c[idx],
                ee[idx],
                ddg[idx],
                ddg_calc[idx],
                diff[idx],
            )
            for idx in worst_idx
        ]
        raise ValueError(
            "{} DDG does not match R*(TEMP+273.15)*ln((1+EE)/(1-EE)); "
            "R={}, atol={}, mismatches={}, worst: {}".format(
                source_csv, DDG_R_KCAL, atol, len(bad_idx), "; ".join(worst)
            )
        )

    return {str(name): float(value) for name, value in zip(data_id, ddg)}


def validate_dataset_ddg_targets(
    dataset_results: list[dict[str, Any]],
    source_ddg: dict[str, float],
    atol: float,
) -> None:
    prefix_keys = {
        "train_test": ("train_name", "train_y"),
        "extra": ("extra_name", "extra_y"),
    }
    for item in dataset_results:
        for prefix, (name_key, y_key) in prefix_keys.items():
            names = item[name_key]
            y = item[y_key]
            missing = [name for name in names if str(name) not in source_ddg]
            if missing:
                raise ValueError("{} {} DATA_ID values are absent from DDG source CSV: {}".format(
                    item["dataset"], prefix, missing[:10]
                ))

            expected = np.asarray([source_ddg[str(name)] for name in names], dtype=float)
            diff = np.abs(np.asarray(y, dtype=float) - expected)
            bad_idx = np.flatnonzero(diff > atol)
            if len(bad_idx):
                worst_idx = bad_idx[np.argsort(diff[bad_idx])[::-1][:10]]
                worst = [
                    "{} y={} source_DDG={} abs_diff={}".format(
                        names[idx],
                        float(y[idx]),
                        float(expected[idx]),
                        float(diff[idx]),
                    )
                    for idx in worst_idx
                ]
                raise ValueError(
                    "{} {} data_y differs from source DDG; atol={}, mismatches={}, worst: {}".format(
                        item["dataset"], prefix, atol, len(bad_idx), "; ".join(worst)
                    )
                )


def validate_feature_label_uniqueness(
    dataset_dir: Path,
    data_x: np.ndarray,
    data_y: np.ndarray,
    labels: list[str],
    names: list[Any],
    source_reaction_keys: dict[str, tuple[Any, ...]] | None,
    duplicate_y_atol: float,
    duplicate_x_decimals: int,
    warning_limit: int,
) -> list[str]:
    if data_x.shape[0] == 0:
        return []
    if data_x.shape[0] != len(data_y) or data_x.shape[0] != len(names):
        raise ValueError("{} duplicate-feature validation row counts do not match".format(dataset_dir))

    rounded_x = np.round(np.asarray(data_x, dtype=float), decimals=duplicate_x_decimals)
    groups: dict[tuple[float, ...], list[int]] = {}
    for idx, row in enumerate(rounded_x):
        groups.setdefault(tuple(row.tolist()), []).append(idx)

    conflicts: list[str] = []
    feature_alias_conflicts: list[str] = []
    consistent_duplicates: list[str] = []
    for indices in groups.values():
        if len(indices) < 2:
            continue
        values = np.asarray(data_y[indices], dtype=float)
        y_range = float(np.max(values) - np.min(values))
        group_names = [str(names[idx]) for idx in indices]
        payload = "DATA_IDs=[{}] | n={} | y_range={:.12g} | {}".format(
            ", ".join(group_names[:warning_limit]),
            len(indices),
            y_range,
            summarize_duplicate_group(group_names, values, warning_limit),
        )
        if y_range > duplicate_y_atol:
            if source_reaction_keys is None:
                conflicts.append(payload)
                continue

            missing_source = [name for name in group_names if name not in source_reaction_keys]
            if missing_source:
                conflicts.append("{} | missing_source_reaction_key={}".format(
                    payload,
                    missing_source[:warning_limit],
                ))
                continue

            by_source_key: dict[tuple[Any, ...], list[int]] = {}
            for idx in indices:
                name = str(names[idx])
                by_source_key.setdefault(source_reaction_keys[name], []).append(idx)

            same_source_conflicts = []
            for source_key, source_indices in by_source_key.items():
                if len(source_indices) < 2:
                    continue
                source_values = np.asarray(data_y[source_indices], dtype=float)
                source_y_range = float(np.max(source_values) - np.min(source_values))
                if source_y_range > duplicate_y_atol:
                    source_names = [str(names[idx]) for idx in source_indices]
                    same_source_conflicts.append(
                        "source_key=[{}] | DATA_IDs=[{}] | y_range={:.12g}".format(
                            ", ".join("{}={}".format(col, value) for col, value in zip(SOURCE_REACTION_KEY_COLUMNS, source_key)),
                            ", ".join(source_names[:warning_limit]),
                            source_y_range,
                        )
                    )

            if same_source_conflicts:
                conflicts.append("{} | same_source_conflicts={}".format(
                    payload,
                    " || ".join(same_source_conflicts[:warning_limit]),
                ))
            else:
                feature_alias_conflicts.append(
                    "{} | source_reaction_keys={} | reason=identical selected features but different source reaction keys".format(
                        payload,
                        len(by_source_key),
                    )
                )
        else:
            consistent_duplicates.append(payload)

    if conflicts:
        examples = " || ".join(conflicts[:warning_limit])
        raise ValueError(
            "{} has feature-label conflicts: identical feature rows rounded to {} decimals map to "
            "different y values beyond atol={}. conflicts={}, examples: {}".format(
                dataset_dir,
                duplicate_x_decimals,
                duplicate_y_atol,
                len(conflicts),
                examples,
            )
        )

    warnings = []
    if feature_alias_conflicts:
        warnings.append(
            "{} has {} feature alias groups: identical feature rows rounded to {} decimals map to different y "
            "values beyond atol={}, but DATA_IDs are different source reaction keys. examples: {}".format(
                dataset_dir,
                len(feature_alias_conflicts),
                duplicate_x_decimals,
                duplicate_y_atol,
                " || ".join(feature_alias_conflicts[:warning_limit]),
            )
        )

    if consistent_duplicates:
        warnings.append(
            "{} has {} duplicated feature rows with consistent y values within atol={}. examples: {}".format(
                dataset_dir,
                len(consistent_duplicates),
                duplicate_y_atol,
                " || ".join(consistent_duplicates[:warning_limit]),
            )
        )
    return warnings


def validate_dataset_dir(
    dataset_dir: Path,
    max_abs_feature_value: float,
    source_reaction_keys: dict[str, tuple[Any, ...]] | None,
    duplicate_y_atol: float,
    duplicate_x_decimals: int,
    warning_limit: int,
) -> dict[str, Any]:
    missing = [
        path.name for path in [
            dataset_dir / "train_test_data_x.npy",
            dataset_dir / "train_test_data_y.npy",
            dataset_dir / "train_test_x_label.pkl",
            dataset_dir / "train_test_data_class.pkl",
            dataset_dir / "train_test_data_name.pkl",
            dataset_dir / "train_test_data_batch.pkl",
            dataset_dir / "extra_data_x.npy",
            dataset_dir / "extra_data_y.npy",
            dataset_dir / "extra_x_label.pkl",
            dataset_dir / "extra_data_class.pkl",
            dataset_dir / "extra_data_name.pkl",
            dataset_dir / "extra_data_batch.pkl",
            dataset_dir / "manifest.json",
            dataset_dir / "selector.json",
            dataset_dir / "selector.pkl",
        ] if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("{} is missing files: {}".format(dataset_dir, ", ".join(missing)))

    train_x = np.load(dataset_dir / "train_test_data_x.npy")
    train_y = np.load(dataset_dir / "train_test_data_y.npy")
    extra_x = np.load(dataset_dir / "extra_data_x.npy")
    extra_y = np.load(dataset_dir / "extra_data_y.npy")
    train_label_payload = load_pickle(dataset_dir / "train_test_x_label.pkl")
    extra_label_payload = load_pickle(dataset_dir / "extra_x_label.pkl")
    train_labels = flatten_labels(train_label_payload)
    extra_labels = flatten_labels(extra_label_payload)
    train_name = load_pickle(dataset_dir / "train_test_data_name.pkl")
    extra_name = load_pickle(dataset_dir / "extra_data_name.pkl")
    train_batch = load_pickle(dataset_dir / "train_test_data_batch.pkl")
    extra_batch = load_pickle(dataset_dir / "extra_data_batch.pkl")
    train_class = load_pickle(dataset_dir / "train_test_data_class.pkl")
    extra_class = load_pickle(dataset_dir / "extra_data_class.pkl")
    manifest = load_json(dataset_dir / "manifest.json")
    selector = load_json(dataset_dir / "selector.json")

    if train_x.shape[0] != len(train_y) or train_x.shape[0] != len(train_name) or train_x.shape[0] != len(train_batch):
        raise ValueError("{} train_test row counts do not match".format(dataset_dir))
    if extra_x.shape[0] != len(extra_y) or extra_x.shape[0] != len(extra_name) or extra_x.shape[0] != len(extra_batch):
        raise ValueError("{} extra row counts do not match".format(dataset_dir))
    if train_x.shape[0] != len(train_class):
        raise ValueError("{} train_test class row count does not match".format(dataset_dir))
    if extra_x.shape[0] != len(extra_class):
        raise ValueError("{} extra class row count does not match".format(dataset_dir))
    if train_x.shape[1] != len(train_labels):
        raise ValueError("{} train label count does not match X columns".format(dataset_dir))
    if extra_x.shape[1] != len(extra_labels):
        raise ValueError("{} extra label count does not match X columns".format(dataset_dir))
    if train_labels != extra_labels:
        raise ValueError("{} train and extra labels differ".format(dataset_dir))
    validate_numeric_values(dataset_dir, "train_test", train_x, train_y, train_labels, train_name, max_abs_feature_value)
    validate_numeric_values(dataset_dir, "extra", extra_x, extra_y, extra_labels, extra_name, max_abs_feature_value)
    if train_labels.count("TEMP") != 1 or train_labels.count("PRESSURE") != 1:
        raise ValueError("{} TEMP/PRESSURE must appear exactly once".format(dataset_dir))
    duplicate_labels = [label for label in train_labels if train_labels.count(label) > 1]
    if duplicate_labels:
        raise ValueError("{} has duplicated feature labels: {}".format(dataset_dir, sorted(set(duplicate_labels))[:10]))
    leakage_labels = [label for label in train_labels if label in LEAKAGE_LABELS]
    if leakage_labels:
        raise ValueError("{} contains target/leakage labels in X: {}".format(dataset_dir, leakage_labels))
    assert_unique(train_name, "train_test DATA_ID", dataset_dir)
    assert_unique(extra_name, "extra DATA_ID", dataset_dir)
    if set(train_name) & set(extra_name):
        raise ValueError("{} train_test and extra DATA_ID sets overlap".format(dataset_dir))

    manifest_counts = manifest.get("row_counts", {})
    if "train_test" in manifest_counts and int(manifest_counts["train_test"]) != int(train_x.shape[0]):
        raise ValueError("{} manifest train_test row count does not match".format(dataset_dir))
    if "extra" in manifest_counts and int(manifest_counts["extra"]) != int(extra_x.shape[0]):
        raise ValueError("{} manifest extra row count does not match".format(dataset_dir))
    if selector.get("selected_label_payload") is not None and selector["selected_label_payload"] != train_label_payload:
        raise ValueError("{} selector selected_label_payload differs from x_label".format(dataset_dir))

    train_ids = manifest.get("train_ids")
    test_ids = manifest.get("test_ids")
    extra_ids = manifest.get("extra_ids")
    if train_ids is not None and test_ids is not None:
        if set(train_ids) & set(test_ids):
            raise ValueError("{} train_ids and test_ids overlap".format(dataset_dir))
        if set(train_ids) | set(test_ids) != set(train_name):
            raise ValueError("{} train_ids + test_ids do not match train_test DATA_ID".format(dataset_dir))
        if selector.get("kind") == "single_feature_selector":
            if selector.get("train_ids") != train_ids or selector.get("test_ids") != test_ids:
                raise ValueError("{} selector train/test ids differ from manifest".format(dataset_dir))
            finite_filter = selector.get("selector", {}).get("finite_filter", "")
            if "train rows only" not in finite_filter:
                raise ValueError("{} selector finite filter was not fitted on train rows only".format(dataset_dir))
    if extra_ids is not None:
        if set(extra_ids) != set(extra_name):
            raise ValueError("{} extra_ids do not match extra DATA_ID".format(dataset_dir))
        if selector.get("kind") == "single_feature_selector" and selector.get("extra_ids") != extra_ids:
            raise ValueError("{} selector extra ids differ from manifest".format(dataset_dir))

    validation_warnings = validate_feature_label_uniqueness(
        dataset_dir=dataset_dir,
        data_x=np.concatenate([train_x, extra_x], axis=0),
        data_y=np.concatenate([train_y, extra_y], axis=0),
        labels=train_labels,
        names=list(train_name) + list(extra_name),
        source_reaction_keys=source_reaction_keys,
        duplicate_y_atol=duplicate_y_atol,
        duplicate_x_decimals=duplicate_x_decimals,
        warning_limit=warning_limit,
    )

    return {
        "dataset": dataset_dir.name,
        "parts": dataset_feature_parts(dataset_dir.name),
        "train_x": train_x,
        "train_y": train_y,
        "train_labels": train_labels,
        "train_label_payload": train_label_payload,
        "train_name": train_name,
        "train_batch": train_batch,
        "train_class": train_class,
        "extra_x": extra_x,
        "extra_y": extra_y,
        "extra_labels": extra_labels,
        "extra_label_payload": extra_label_payload,
        "extra_name": extra_name,
        "extra_batch": extra_batch,
        "extra_class": extra_class,
        "train_test_rows": int(train_x.shape[0]),
        "extra_rows": int(extra_x.shape[0]),
        "features": int(train_x.shape[1]),
        "warnings": validation_warnings,
    }


def validate_cross_dataset_consistency(dataset_results: list[dict[str, Any]]) -> None:
    by_name = {item["dataset"]: item for item in dataset_results}
    single_feature_names = sorted(FEATURE_TYPES & set(by_name.keys()))
    if not single_feature_names:
        return

    reference = by_name[single_feature_names[0]]
    for item in dataset_results:
        assert_same(reference["train_name"], item["train_name"], "train_test DATA_ID order", Path(item["dataset"]))
        assert_same(reference["extra_name"], item["extra_name"], "extra DATA_ID order", Path(item["dataset"]))
        assert_same(reference["train_y"], item["train_y"], "train_test y", Path(item["dataset"]))
        assert_same(reference["extra_y"], item["extra_y"], "extra y", Path(item["dataset"]))
        assert_same(reference["train_batch"], item["train_batch"], "train_test batch", Path(item["dataset"]))
        assert_same(reference["extra_batch"], item["extra_batch"], "extra batch", Path(item["dataset"]))
        assert_same(reference["train_class"], item["train_class"], "train_test class", Path(item["dataset"]))
        assert_same(reference["extra_class"], item["extra_class"], "extra class", Path(item["dataset"]))

    for item in dataset_results:
        parts = item["parts"]
        if len(parts) == 1:
            continue
        if any(part not in by_name for part in parts):
            continue

        x_parts_train = []
        x_parts_extra = []
        label_payload: dict[str, list[str]] = {}
        for idx, part in enumerate(parts, start=1):
            component = by_name[part]
            component_train_x = component["train_x"]
            component_extra_x = component["extra_x"]
            component_labels = component["train_labels"]
            if idx > 1:
                component_train_x, component_labels = drop_duplicate_conditions(component_train_x, component_labels)
                component_extra_x, _ = drop_duplicate_conditions(component_extra_x, component["extra_labels"])
            x_parts_train.append(component_train_x)
            x_parts_extra.append(component_extra_x)
            label_payload["label{}_{}".format(idx, part)] = component_labels

        expected_train_x = np.concatenate(x_parts_train, axis=1)
        expected_extra_x = np.concatenate(x_parts_extra, axis=1)
        expected_labels = flatten_labels(label_payload)
        dataset_path = Path(item["dataset"])
        assert_same(expected_labels, item["train_labels"], "combined feature labels", dataset_path)
        assert_same(expected_train_x, item["train_x"], "combined train_test X", dataset_path)
        assert_same(expected_extra_x, item["extra_x"], "combined extra X", dataset_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate AAReact ML datasets generated under Data_All/3_data_for_train.")
    parser.add_argument("--base", type=Path, default=Path(__file__).resolve().parents[1] / "Data_All")
    parser.add_argument("--target", choices=["ee", "ddg"], required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--split-name", default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--source-csv", type=Path, default=None, help="Canonical source CSV for target validation.")
    parser.add_argument("--ddg-atol", type=float, default=DEFAULT_DDG_ATOL)
    parser.add_argument(
        "--max-abs-feature-value",
        type=float,
        default=DEFAULT_MAX_ABS_FEATURE_VALUE,
        help="Fail when any feature absolute value exceeds this threshold.",
    )
    parser.add_argument(
        "--source-duplicate-y-atol",
        type=float,
        default=DEFAULT_DUPLICATE_Y_ATOL,
        help="Fail when identical source reaction keys have target range above this value.",
    )
    parser.add_argument(
        "--feature-duplicate-y-atol",
        type=float,
        default=DEFAULT_DUPLICATE_Y_ATOL,
        help="Fail when identical feature rows have target range above this value.",
    )
    parser.add_argument(
        "--feature-duplicate-x-decimals",
        type=int,
        default=DEFAULT_DUPLICATE_X_DECIMALS,
        help="Decimals used when grouping feature rows for duplicate-X validation.",
    )
    parser.add_argument(
        "--duplicate-warning-limit",
        type=int,
        default=DEFAULT_DUPLICATE_WARNING_LIMIT,
        help="Maximum duplicate examples shown in errors or warnings.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_name = args.split_name or split_tag(args.seed, args.test_size)
    source_ddg = None
    source_csv = args.source_csv or (args.base / "full_data_436-20260723.csv")
    source_reaction_keys = load_source_reaction_key_map(source_csv)
    validation_warnings: list[str] = []
    validation_warnings.extend(
        validate_source_label_uniqueness(
            source_csv=source_csv,
            target=args.target,
            duplicate_y_atol=args.source_duplicate_y_atol,
            warning_limit=args.duplicate_warning_limit,
        )
    )
    if args.target == "ddg":
        source_ddg = load_and_validate_ddg_source(source_csv, args.ddg_atol)

    root = args.base / "3_data_for_train" / args.target / split_name
    if not root.exists():
        raise FileNotFoundError("Dataset root does not exist: {}".format(root))
    dataset_dirs = [root / item for item in args.datasets] if args.datasets else sorted(path for path in root.iterdir() if path.is_dir())
    results = [
        validate_dataset_dir(
            dataset_dir=dataset_dir,
            max_abs_feature_value=args.max_abs_feature_value,
            source_reaction_keys=source_reaction_keys,
            duplicate_y_atol=args.feature_duplicate_y_atol,
            duplicate_x_decimals=args.feature_duplicate_x_decimals,
            warning_limit=args.duplicate_warning_limit,
        )
        for dataset_dir in dataset_dirs
    ]
    validate_cross_dataset_consistency(results)
    if source_ddg is not None:
        validate_dataset_ddg_targets(results, source_ddg, args.ddg_atol)
    for item in results:
        for warning in item.get("warnings", []):
            validation_warnings.append("{}: {}".format(item["dataset"], warning))
    for warning in validation_warnings:
        print("[WARN] {}".format(warning), file=sys.stderr)

    checks = [
        "required_files",
        "row_and_label_shapes",
        "finite_values",
        "max_abs_feature_values",
        "source_label_uniqueness",
        "feature_label_uniqueness",
        "feature_alias_warning_for_distinct_source_reactions",
        "temp_pressure_once",
        "target_leakage_labels_absent",
        "train_test_extra_id_disjoint",
        "selector_manifest_consistency",
        "cross_dataset_metadata_consistency",
        "combined_feature_value_consistency",
    ]
    if args.target == "ddg":
        checks.extend([
            "source_ddg_formula_kelvin",
            "dataset_y_matches_source_ddg",
        ])

    summaries = [
        {
            "dataset": item["dataset"],
            "train_test_rows": item["train_test_rows"],
            "extra_rows": item["extra_rows"],
            "features": item["features"],
        }
        for item in results
    ]
    print(json.dumps({
        "target": args.target,
        "split": split_name,
        "source_csv": str(source_csv),
        "max_abs_feature_value": args.max_abs_feature_value,
        "source_duplicate_y_atol": args.source_duplicate_y_atol,
        "feature_duplicate_y_atol": args.feature_duplicate_y_atol,
        "feature_duplicate_x_decimals": args.feature_duplicate_x_decimals,
        "checks": checks,
        "warnings": validation_warnings,
        "datasets": summaries,
    }, indent=2))


if __name__ == "__main__":
    main()
