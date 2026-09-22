import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RAW_FEATURE_FILES = {
    "rdkit": "rdkit_desc_features.csv",
    "soap": "soap_features.csv",
    "xtb": "xtb_features.csv",
    "acsf": "acsf_features.csv",
}


def load_pickle(fp: str | Path) -> Any:
    with open(fp, "rb") as f:
        return pickle.load(f)


def save_pickle(fp: str | Path, value: Any) -> None:
    with open(fp, "wb") as f:
        pickle.dump(value, f)


def label_order(label_key: str) -> int:
    match = re.match(r"label(\d+)_", label_key)
    if match is None:
        raise ValueError("Invalid feature label key: {}".format(label_key))
    return int(match.group(1))


def feature_name_from_label_key(label_key: str) -> str:
    if "_" not in label_key:
        raise ValueError("Invalid feature label key: {}".format(label_key))
    return label_key.split("_", 1)[1]


def ordered_label_items(label_payload: dict[str, list[str]]) -> list[tuple[str, list[str]]]:
    if not isinstance(label_payload, dict):
        raise TypeError("Feature labels must be a dict, got {}".format(type(label_payload).__name__))
    return sorted(label_payload.items(), key=lambda item: label_order(item[0]))


def flatten_label_payload(label_payload: dict[str, list[str]]) -> list[str]:
    labels: list[str] = []
    for _, group_labels in ordered_label_items(label_payload):
        labels.extend(group_labels)
    return labels


def read_ids_file(fp: str | Path) -> list[str]:
    ids = []
    for line in Path(fp).read_text().splitlines():
        item = line.strip()
        if not item or item == "DATA_ID":
            continue
        ids.append(item.split(",")[0].strip())
    return ids


def select_rows(df: pd.DataFrame, ids: list[str] | None, batches: list[int] | None, source_name: str) -> pd.DataFrame:
    if ids is None and batches is None:
        raise ValueError("Use ids or batches to select rows for {}".format(source_name))

    if "DATA_ID" not in df.columns:
        raise ValueError("{} is missing DATA_ID".format(source_name))
    if "BATCH" not in df.columns:
        raise ValueError("{} is missing BATCH".format(source_name))

    df = df.copy()
    df["DATA_ID"] = df["DATA_ID"].astype(str)
    if ids is not None:
        if df["DATA_ID"].duplicated().any():
            duplicated = df["DATA_ID"][df["DATA_ID"].duplicated()].tolist()
            raise ValueError("{} has duplicated DATA_ID values: {}".format(source_name, duplicated[:10]))
        indexed = df.set_index("DATA_ID", drop=False)
        missing = [item for item in ids if item not in indexed.index]
        if missing:
            raise ValueError("{} is missing requested DATA_ID values: {}".format(source_name, missing[:10]))
        selected = indexed.loc[ids].reset_index(drop=True)
        if batches is not None:
            batch_values = pd.to_numeric(selected["BATCH"], errors="raise").astype(int)
            bad_ids = selected.loc[~batch_values.isin(batches), "DATA_ID"].tolist()
            if bad_ids:
                raise ValueError("{} selected ids are outside requested batches: {}".format(source_name, bad_ids[:10]))
        return selected

    batch_values = pd.to_numeric(df["BATCH"], errors="raise").astype(int)
    return df.loc[batch_values.isin(batches)].reset_index(drop=True)


def numeric_columns(df: pd.DataFrame, labels: list[str], source_name: str) -> np.ndarray:
    missing = [label for label in labels if label not in df.columns]
    if missing:
        raise ValueError("{} is missing selected feature columns: {}".format(source_name, missing[:10]))
    return df.loc[:, labels].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def nonfinite_labels(data_x: np.ndarray, labels: list[str]) -> list[str]:
    if data_x.size == 0:
        return []
    finite_mask = np.all(np.isfinite(data_x), axis=0)
    return [label for label, ok in zip(labels, finite_mask) if not ok]


def transform_label_payload_from_raw(
    raw_feature_dir: str | Path,
    label_payload: dict[str, list[str]],
    target_col: str,
    ids: list[str] | None = None,
    batches: list[int] | None = None,
    require_target: bool = False,
    allow_nonfinite: bool = False,
) -> dict[str, Any]:
    raw_feature_dir = Path(raw_feature_dir)
    x_parts = []
    source_files: dict[str, str] = {}
    selected_ids: list[str] | None = ids
    metadata: dict[str, Any] | None = None

    for label_key, labels in ordered_label_items(label_payload):
        feature_name = feature_name_from_label_key(label_key)
        if feature_name not in RAW_FEATURE_FILES:
            raise ValueError("Unsupported feature type in labels: {}".format(feature_name))
        source_fp = raw_feature_dir / RAW_FEATURE_FILES[feature_name]
        if not source_fp.exists():
            raise FileNotFoundError("Raw feature CSV does not exist: {}".format(source_fp))

        df = pd.read_csv(source_fp, low_memory=False)
        selected = select_rows(df, selected_ids, batches if selected_ids is None else None, str(source_fp))
        if selected_ids is None:
            selected_ids = selected["DATA_ID"].astype(str).tolist()

        x_part = numeric_columns(selected, labels, str(source_fp))
        bad_labels = nonfinite_labels(x_part, labels)
        if bad_labels and not allow_nonfinite:
            raise ValueError("{} has non-finite values in selected features: {}".format(source_fp, bad_labels[:10]))

        x_parts.append(x_part)
        source_files[feature_name] = str(source_fp)

        if metadata is None:
            if target_col in selected.columns:
                data_y = pd.to_numeric(selected[target_col], errors="coerce").to_numpy(dtype=float)
                if require_target and not np.isfinite(data_y).all():
                    bad_ids = selected.loc[~np.isfinite(data_y), "DATA_ID"].astype(str).tolist()
                    raise ValueError("{} has missing target {} for ids: {}".format(source_fp, target_col, bad_ids[:10]))
            elif require_target:
                raise ValueError("{} is missing target column {}".format(source_fp, target_col))
            else:
                data_y = np.full(len(selected), np.nan, dtype=float)

            metadata = {
                "data_y": data_y,
                "data_class": pd.to_numeric(selected["CLASS"], errors="coerce").astype("Int64").tolist(),
                "data_name": selected["DATA_ID"].astype(str).tolist(),
                "data_batch": pd.to_numeric(selected["BATCH"], errors="raise").astype(int).tolist(),
            }

    if metadata is None:
        raise ValueError("No feature labels were provided.")

    data_x = np.concatenate(x_parts, axis=1) if x_parts else np.empty((len(metadata["data_name"]), 0))
    return {
        "data_x": data_x,
        "data_y": metadata["data_y"],
        "x_label": label_payload,
        "data_class": metadata["data_class"],
        "data_name": metadata["data_name"],
        "data_batch": metadata["data_batch"],
        "source_files": source_files,
    }


def save_dataset_files(out_dir: str | Path, prefix: str, transformed: dict[str, Any]) -> None:
    out_dir = Path(out_dir)
    np.save(out_dir / "{}_data_x.npy".format(prefix), transformed["data_x"])
    np.save(out_dir / "{}_data_y.npy".format(prefix), transformed["data_y"])
    save_pickle(out_dir / "{}_x_label.pkl".format(prefix), transformed["x_label"])
    save_pickle(out_dir / "{}_data_class.pkl".format(prefix), transformed["data_class"])
    save_pickle(out_dir / "{}_data_name.pkl".format(prefix), transformed["data_name"])
    save_pickle(out_dir / "{}_data_batch.pkl".format(prefix), transformed["data_batch"])
