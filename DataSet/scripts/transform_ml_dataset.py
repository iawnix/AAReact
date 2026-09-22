#!/usr/bin/env python3
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

from config.constants import normalize_target, target_column
from util.feature_transform import (
    load_pickle,
    read_ids_file,
    save_dataset_files,
    transform_label_payload_from_raw,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply saved feature selectors to new test/extra rows without refitting feature selection."
    )
    parser.add_argument("--base", type=Path, default=Path("/home/iaw/DATA2/AAReact/DataSet/Data_All"))
    parser.add_argument("--target", choices=["ee", "ddg"], required=True)
    parser.add_argument("--split-name", default="seed_1_test_0-2")
    parser.add_argument("--raw-feature-dir", type=Path, default=None)
    parser.add_argument("--datasets", nargs="*", default=None, help="Dataset names such as rdkit or rdkit_soap_xtb.")
    parser.add_argument("--prefix", required=True, help="Output file prefix, e.g. extra_batch2 or test_external.")
    parser.add_argument("--batches", nargs="*", type=int, default=None, help="BATCH values to transform.")
    parser.add_argument("--ids-file", type=Path, default=None, help="Optional DATA_ID list, one id per line or first CSV column.")
    parser.add_argument("--require-target", action="store_true", help="Require finite target values in the selected rows.")
    parser.add_argument("--allow-nonfinite", action="store_true", help="Allow non-finite selected feature values.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files with the same prefix.")
    return parser.parse_args()


def load_label_payload(dataset_dir: Path) -> tuple[dict[str, list[str]], str]:
    selector_fp = dataset_dir / "selector.json"
    if selector_fp.exists():
        selector = json.loads(selector_fp.read_text())
        return selector["selected_label_payload"], str(selector_fp)
    label_fp = dataset_dir / "train_test_x_label.pkl"
    if not label_fp.exists():
        raise FileNotFoundError("Missing selector.json or train_test_x_label.pkl in {}".format(dataset_dir))
    return load_pickle(label_fp), str(label_fp)


def selected_dataset_dirs(root: Path, datasets: list[str] | None) -> list[Path]:
    if datasets:
        dirs = [root / name for name in datasets]
    else:
        dirs = sorted(path for path in root.iterdir() if path.is_dir())
    missing = [str(path) for path in dirs if not path.exists()]
    if missing:
        raise FileNotFoundError("Dataset directories do not exist: {}".format(missing[:10]))
    return dirs


def assert_can_write(dataset_dir: Path, prefix: str, overwrite: bool) -> None:
    suffixes = [
        "data_x.npy",
        "data_y.npy",
        "x_label.pkl",
        "data_class.pkl",
        "data_name.pkl",
        "data_batch.pkl",
        "manifest.json",
    ]
    existing = [dataset_dir / "{}_{}".format(prefix, suffix) for suffix in suffixes]
    existing = [path for path in existing if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Output files already exist for prefix {}: {}".format(prefix, existing[:3]))


def write_transform_manifest(
    dataset_dir: Path,
    prefix: str,
    target: str,
    target_col: str,
    split_name: str,
    selector_source: str,
    transformed: dict[str, Any],
    batches: list[int] | None,
    ids_file: Path | None,
    require_target: bool,
    allow_nonfinite: bool,
) -> None:
    manifest = {
        "version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "kind": "selector_transform",
        "target": target,
        "target_column": target_col,
        "split_name": split_name,
        "dataset": dataset_dir.name,
        "prefix": prefix,
        "selector_source": selector_source,
        "batches": batches,
        "ids_file": str(ids_file) if ids_file is not None else None,
        "require_target": require_target,
        "allow_nonfinite": allow_nonfinite,
        "row_count": int(transformed["data_x"].shape[0]),
        "feature_count": int(transformed["data_x"].shape[1]),
        "data_ids": transformed["data_name"],
        "source_files": transformed["source_files"],
    }
    (dataset_dir / "{}_manifest.json".format(prefix)).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )


def main() -> None:
    args = parse_args()
    target = normalize_target(args.target)
    target_col = target_column(target)
    ids = read_ids_file(args.ids_file) if args.ids_file is not None else None
    batches = args.batches if args.batches else None
    if ids is None and batches is None:
        raise ValueError("Use --batches or --ids-file to select new rows.")

    raw_feature_dir = args.raw_feature_dir or args.base / "2_raw_features"
    root = args.base / "3_data_for_train" / target / args.split_name
    if not root.exists():
        raise FileNotFoundError("Dataset root does not exist: {}".format(root))

    summaries = []
    for dataset_dir in selected_dataset_dirs(root, args.datasets):
        label_payload, selector_source = load_label_payload(dataset_dir)
        assert_can_write(dataset_dir, args.prefix, args.overwrite)
        transformed = transform_label_payload_from_raw(
            raw_feature_dir=raw_feature_dir,
            label_payload=label_payload,
            target_col=target_col,
            ids=ids,
            batches=batches,
            require_target=args.require_target,
            allow_nonfinite=args.allow_nonfinite,
        )
        save_dataset_files(dataset_dir, args.prefix, transformed)
        write_transform_manifest(
            dataset_dir=dataset_dir,
            prefix=args.prefix,
            target=target,
            target_col=target_col,
            split_name=args.split_name,
            selector_source=selector_source,
            transformed=transformed,
            batches=batches,
            ids_file=args.ids_file,
            require_target=args.require_target,
            allow_nonfinite=args.allow_nonfinite,
        )
        summaries.append({
            "dataset": dataset_dir.name,
            "rows": int(transformed["data_x"].shape[0]),
            "features": int(transformed["data_x"].shape[1]),
        })

    print(json.dumps({"target": target, "split": args.split_name, "prefix": args.prefix, "outputs": summaries}, indent=2))


if __name__ == "__main__":
    main()
