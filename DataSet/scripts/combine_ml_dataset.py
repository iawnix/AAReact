#!/usr/bin/env python3
import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


FEATURE_COMBINATIONS = [
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
ALWAYS_KEEP_FEATURES = ["TEMP", "PRESSURE"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine target-specific single-feature ML datasets.")
    parser.add_argument("--base", type=Path, default=Path("/home/iaw/DATA2/AAReact/DataSet/Data_All"))
    parser.add_argument("--target", choices=["ee", "ddg"], required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--combinations", nargs="*", default=None, help="Optional combinations such as rdkit_soap.")
    return parser.parse_args()


def split_tag(seed: int, test_size: float) -> str:
    return "seed_{}_test_{}".format(seed, str(test_size).replace(".", "-"))


def selected_combinations(args: argparse.Namespace) -> list[tuple[str, ...]]:
    if args.combinations is None:
        return FEATURE_COMBINATIONS
    return [tuple(item.split("_")) for item in args.combinations]


def load_pickle(fp: Path) -> Any:
    with open(fp, "rb") as f:
        return pickle.load(f)


def save_pickle(fp: Path, value: Any) -> None:
    with open(fp, "wb") as f:
        pickle.dump(value, f)


def load_component(root: Path, feature: str, prefix: str) -> dict[str, Any]:
    feature_dir = root / feature
    manifest_fp = feature_dir / "manifest.json"
    manifest_data = json.loads(manifest_fp.read_text()) if manifest_fp.exists() else {}
    x = np.load(feature_dir / "{}_data_x.npy".format(prefix))
    y = np.load(feature_dir / "{}_data_y.npy".format(prefix))
    labels = load_pickle(feature_dir / "{}_x_label.pkl".format(prefix))["label1_{}".format(feature)]
    data_class = load_pickle(feature_dir / "{}_data_class.pkl".format(prefix))
    data_name = load_pickle(feature_dir / "{}_data_name.pkl".format(prefix))
    data_batch = load_pickle(feature_dir / "{}_data_batch.pkl".format(prefix))
    return {
        "feature": feature,
        "x": x,
        "y": y,
        "labels": labels,
        "data_class": data_class,
        "data_name": data_name,
        "data_batch": data_batch,
        "manifest": manifest_fp,
        "manifest_data": manifest_data,
    }


def assert_same(reference: Any, current: Any, name: str, feature: str, prefix: str) -> None:
    if isinstance(reference, np.ndarray):
        ok = np.array_equal(reference, current, equal_nan=True)
    else:
        ok = reference == current
    if not ok:
        raise ValueError("{} mismatch for {} {}".format(name, feature, prefix))


def drop_duplicate_conditions(component: dict[str, Any]) -> tuple[np.ndarray, list[str], list[str]]:
    keep_idx = []
    dropped = []
    for idx, label in enumerate(component["labels"]):
        if label in ALWAYS_KEEP_FEATURES:
            dropped.append(label)
        else:
            keep_idx.append(idx)
    return component["x"][:, keep_idx], [component["labels"][i] for i in keep_idx], dropped


def save_dataset(out_dir: Path, prefix: str, data_x: np.ndarray, data_y: np.ndarray, labels: dict[str, list[str]],
                 data_class: list[Any], data_name: list[str], data_batch: list[int]) -> None:
    np.save(out_dir / "{}_data_x.npy".format(prefix), data_x)
    np.save(out_dir / "{}_data_y.npy".format(prefix), data_y)
    save_pickle(out_dir / "{}_x_label.pkl".format(prefix), labels)
    save_pickle(out_dir / "{}_data_class.pkl".format(prefix), data_class)
    save_pickle(out_dir / "{}_data_name.pkl".format(prefix), data_name)
    save_pickle(out_dir / "{}_data_batch.pkl".format(prefix), data_batch)


def save_selector_artifact(out_dir: Path, artifact: dict[str, Any]) -> None:
    (out_dir / "selector.json").write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n")
    save_pickle(out_dir / "selector.pkl", artifact)


def combine_prefix(root: Path, combo: tuple[str, ...], prefix: str) -> dict[str, Any]:
    components = [load_component(root, feature, prefix) for feature in combo]
    first = components[0]
    x_parts = [first["x"]]
    labels = {"label1_{}".format(first["feature"]): first["labels"]}
    dropped_conditions = {}

    for idx, component in enumerate(components[1:], start=2):
        assert_same(first["y"], component["y"], "data_y", component["feature"], prefix)
        assert_same(first["data_class"], component["data_class"], "data_class", component["feature"], prefix)
        assert_same(first["data_name"], component["data_name"], "data_name", component["feature"], prefix)
        assert_same(first["data_batch"], component["data_batch"], "data_batch", component["feature"], prefix)
        x_component, labels_component, dropped = drop_duplicate_conditions(component)
        x_parts.append(x_component)
        labels["label{}_{}".format(idx, component["feature"])] = labels_component
        dropped_conditions[component["feature"]] = dropped

    source_target_csvs = sorted({
        str(component["manifest_data"].get("source_target_csv"))
        for component in components
        if component["manifest_data"].get("source_target_csv")
    })

    return {
        "x": np.concatenate(x_parts, axis=1),
        "y": first["y"],
        "labels": labels,
        "data_class": first["data_class"],
        "data_name": first["data_name"],
        "data_batch": first["data_batch"],
        "component_manifests": [str(component["manifest"]) for component in components],
        "component_selectors": [str(component["manifest"].with_name("selector.json")) for component in components],
        "source_target_csv": source_target_csvs[0] if len(source_target_csvs) == 1 else None,
        "component_source_target_csvs": source_target_csvs,
        "dropped_duplicate_conditions": dropped_conditions,
    }


def combine_one(root: Path, combo: tuple[str, ...], target: str, seed: int, test_size: float) -> dict[str, Any]:
    combo_name = "_".join(combo)
    out_dir = root / combo_name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_test = combine_prefix(root, combo, "train_test")
    extra = combine_prefix(root, combo, "extra")
    save_dataset(
        out_dir,
        "train_test",
        train_test["x"],
        train_test["y"],
        train_test["labels"],
        train_test["data_class"],
        train_test["data_name"],
        train_test["data_batch"],
    )
    save_dataset(
        out_dir,
        "extra",
        extra["x"],
        extra["y"],
        extra["labels"],
        extra["data_class"],
        extra["data_name"],
        extra["data_batch"],
    )

    manifest = {
        "version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "target": target,
        "seed": seed,
        "test_size": test_size,
        "combination": list(combo),
        "source_target_csv": train_test["source_target_csv"],
        "component_source_target_csvs": train_test["component_source_target_csvs"],
        "component_manifests": train_test["component_manifests"],
        "component_selectors": train_test["component_selectors"],
        "row_counts": {
            "train_test": int(train_test["x"].shape[0]),
            "extra": int(extra["x"].shape[0]),
        },
        "feature_counts": {
            "train_test": int(train_test["x"].shape[1]),
            "extra": int(extra["x"].shape[1]),
        },
        "dropped_duplicate_conditions": train_test["dropped_duplicate_conditions"],
    }
    selector_artifact = {
        "version": 1,
        "kind": "combined_feature_selector",
        "created_at": manifest["created_at"],
        "target": target,
        "seed": seed,
        "test_size": test_size,
        "combination": list(combo),
        "source_target_csv": train_test["source_target_csv"],
        "component_source_target_csvs": train_test["component_source_target_csvs"],
        "selected_label_payload": train_test["labels"],
        "component_manifests": train_test["component_manifests"],
        "component_selectors": train_test["component_selectors"],
        "dropped_duplicate_conditions": train_test["dropped_duplicate_conditions"],
    }
    save_selector_artifact(out_dir, selector_artifact)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return {"combination": combo_name, "out_dir": str(out_dir), "feature_count": manifest["feature_counts"]["train_test"]}


def main() -> None:
    args = parse_args()
    root = args.base / "3_data_for_train" / args.target / split_tag(args.seed, args.test_size)
    if not root.exists():
        raise FileNotFoundError("Dataset root does not exist: {}".format(root))
    summaries = [combine_one(root, combo, args.target, args.seed, args.test_size) for combo in selected_combinations(args)]
    print(json.dumps({"target": args.target, "split": split_tag(args.seed, args.test_size), "outputs": summaries}, indent=2))


if __name__ == "__main__":
    main()
