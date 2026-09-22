#!/usr/bin/env python3
"""Print block-style AAReact ML metrics from train.csv."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Iterable

import pandas as pd


TRAIN_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = TRAIN_DIR / "train.csv"
DEFAULT_OUTPUT = TRAIN_DIR / "Analysis" / "train_metrics" / "model_metrics_table.csv"
PRIMARY_SEED = 1
PRIMARY_TEST_SIZE = 0.2

METRIC_COLUMNS = [
    "R2_train",
    "R2_test",
    "R_train",
    "R_test",
    "Spearmanr_train",
    "Spearmanr_test",
    "MSE_train",
    "MSE_test",
    "MAE_train",
    "MAE_test",
    "RMSE_train",
    "RMSE_test",
]

KEY_COLUMNS = [
    "marker",
    "seed",
    "test_size",
    "RMSE_test",
    "R2_test",
    "Spearmanr_test",
    "MAE_test",
    "RMSE_train",
    "R2_train",
]

FULL_COLUMNS = ["marker", "seed", "test_size", *METRIC_COLUMNS]

SUMMARY_COLUMNS = [
    "rank",
    "model_name",
    "descript",
    "n_splits",
    "RMSE_test_mean",
    "RMSE_test_std",
    "RMSE_test_min",
    "RMSE_test_max",
    "R2_test_mean",
    "Spearmanr_test_mean",
]

PRIMARY_COLUMNS = [
    "rank",
    "model_name",
    "descript",
    "seed",
    "test_size",
    "RMSE_test",
    "MAE_test",
    "R2_test",
    "Spearmanr_test",
    "RMSE_train",
]

STABILITY_COLUMNS = [
    "primary_marker",
    "test_size",
    "n_seeds",
    "RMSE_test_primary_seed",
    "RMSE_test_mean",
    "RMSE_test_std",
    "RMSE_test_max",
    "R2_test_primary_seed",
    "R2_test_mean",
    "Spearmanr_test_mean",
]

MODEL_STYLES = {
    "rf": "1;32",
    "xgb": "1;33",
    "lgb": "1;34",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show AAReact ML metrics grouped by model and descriptor."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input train metrics CSV.")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT, help="Optional saved table path.")
    parser.add_argument(
        "--view",
        choices=["primary", "blocks", "summary", "flat"],
        default="primary",
        help=(
            "primary ranks model/descriptor choices at the primary seed and test size; "
            "blocks prints one model/descriptor block; summary prints grouped means; flat prints rows."
        ),
    )
    parser.add_argument(
        "--sort-blocks",
        choices=["rmse_mean", "model_descriptor"],
        default="rmse_mean",
        help="Block order for --view blocks.",
    )
    parser.add_argument(
        "--sort-rows",
        choices=["seed_test", "rmse_test"],
        default="seed_test",
        help="Row order inside each block.",
    )
    parser.add_argument("--top-blocks", type=int, default=0, help="Show only top N blocks; 0 means all blocks.")
    parser.add_argument("--top", type=int, default=0, help="Show only top N rows in --view flat; 0 means all rows.")
    parser.add_argument(
        "--primary-seed",
        type=int,
        default=PRIMARY_SEED,
        help="Seed used for primary model selection in --view primary.",
    )
    parser.add_argument(
        "--primary-test-size",
        type=float,
        default=PRIMARY_TEST_SIZE,
        help="Test size used for primary model selection in --view primary.",
    )
    parser.add_argument("--full", action="store_true", help="Show all train/test metrics in each row.")
    parser.add_argument("--save", action="store_true", help="Write displayed data to --output-csv.")
    parser.add_argument("--model-name", choices=["lgb", "xgb", "rf"], default=None, help="Filter by model.")
    parser.add_argument("--descriptor", default=None, help="Filter by descriptor name.")
    parser.add_argument("--seed", type=int, default=None, help="Filter by seed.")
    parser.add_argument("--test-size", type=float, default=None, help="Filter by test size.")
    parser.add_argument(
        "--color",
        choices=["auto", "always", "never"],
        default="auto",
        help="Color terminal output. auto disables color when stdout is not a TTY.",
    )
    return parser.parse_args()


def split_sort_key(split_name: object) -> tuple[int, float]:
    parts = str(split_name).split("_")
    if len(parts) >= 4 and parts[0] == "seed" and parts[2] == "test":
        return int(parts[1]), float(parts[3].replace("-", "."))
    return 999999, 999999.0


def require_columns(df: pd.DataFrame, columns: Iterable[str], input_fp: Path) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError("{} is missing required columns: {}".format(input_fp, ", ".join(missing)))


def load_metrics(input_fp: Path) -> pd.DataFrame:
    if not input_fp.exists():
        raise FileNotFoundError(
            "Input CSV does not exist: {}. Run train/2_extract_train_model_metric.sh first.".format(input_fp)
        )

    df = pd.read_csv(input_fp)
    require_columns(df, ["model", "descriptor", "split", *METRIC_COLUMNS], input_fp)

    if "target" not in df.columns:
        df["target"] = "unknown"
    if "hyper_split" not in df.columns:
        df["hyper_split"] = "unknown"
    if "search_method" not in df.columns:
        df["search_method"] = "legacy"

    if "seed" not in df.columns or "test_size" not in df.columns:
        parsed = df["split"].map(split_sort_key)
        df["seed"] = [item[0] for item in parsed]
        df["test_size"] = [item[1] for item in parsed]

    for col in METRIC_COLUMNS + ["seed", "test_size"]:
        df[col] = pd.to_numeric(df[col], errors="raise")

    df = df.copy()
    df["model_name"] = df["model"].astype(str)
    df["descript"] = df["descriptor"].astype(str)
    return df


def is_close(series: pd.Series, value: float) -> pd.Series:
    return (series.astype(float) - float(value)).abs() < 1.0e-12


def apply_filters(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df
    if args.model_name is not None:
        out = out[out["model_name"] == args.model_name]
    if args.descriptor is not None:
        out = out[out["descript"] == args.descriptor]
    if args.seed is not None:
        out = out[out["seed"] == args.seed]
    if args.test_size is not None:
        out = out[is_close(out["test_size"], args.test_size)]
    if out.empty:
        raise ValueError("No rows match the requested filters.")
    return out.copy()


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["target", "hyper_split", "search_method", "model_name", "descript"]
    summary = (
        df.groupby(keys, dropna=False)
        .agg(
            n_splits=("split", "nunique"),
            RMSE_test_mean=("RMSE_test", "mean"),
            RMSE_test_std=("RMSE_test", "std"),
            RMSE_test_min=("RMSE_test", "min"),
            RMSE_test_max=("RMSE_test", "max"),
            R2_test_mean=("R2_test", "mean"),
            Spearmanr_test_mean=("Spearmanr_test", "mean"),
        )
        .reset_index()
    )
    summary = summary.sort_values(["RMSE_test_mean", "RMSE_test_std", "RMSE_test_max"], ascending=True)
    summary.insert(0, "rank", range(1, len(summary) + 1))
    return summary


def build_primary_summary(df: pd.DataFrame, primary_seed: int, primary_test_size: float) -> pd.DataFrame:
    primary = df[(df["seed"] == int(primary_seed)) & is_close(df["test_size"], primary_test_size)].copy()
    if primary.empty:
        raise ValueError(
            "No rows found for primary seed={} and test_size={:.4f}. Available splits: {}".format(
                primary_seed,
                primary_test_size,
                ", ".join(
                    "seed_{}_test_{:.4f}".format(int(row.seed), float(row.test_size))
                    for row in df[["seed", "test_size"]].drop_duplicates().sort_values(["seed", "test_size"]).itertuples()
                ),
            )
        )

    columns = [
        "target",
        "hyper_split",
        "search_method",
        "model_name",
        "descript",
        "seed",
        "test_size",
        "RMSE_test",
        "MAE_test",
        "R2_test",
        "Spearmanr_test",
        "RMSE_train",
    ]
    summary = primary[columns].copy()
    summary = summary.sort_values(["RMSE_test", "MAE_test", "R2_test"], ascending=[True, True, False])
    summary.insert(0, "rank", range(1, len(summary) + 1))
    return summary


def build_stability_rows(
    df: pd.DataFrame,
    primary_summary: pd.DataFrame,
    primary_seed: int,
    primary_test_size: float,
    top_blocks: int,
) -> list[tuple[pd.Series, pd.DataFrame]]:
    selected = primary_summary
    if top_blocks and top_blocks > 0:
        selected = selected.head(top_blocks)

    output: list[tuple[pd.Series, pd.DataFrame]] = []
    for _, item in selected.iterrows():
        mask = (df["model_name"] == item["model_name"]) & (df["descript"] == item["descript"])
        candidate_rows = df[mask].copy()
        block = (
            candidate_rows
            .groupby("test_size", dropna=False)
            .agg(
                n_seeds=("seed", "nunique"),
                RMSE_test_mean=("RMSE_test", "mean"),
                RMSE_test_std=("RMSE_test", "std"),
                RMSE_test_max=("RMSE_test", "max"),
                R2_test_mean=("R2_test", "mean"),
                Spearmanr_test_mean=("Spearmanr_test", "mean"),
            )
            .reset_index()
            .sort_values("test_size", ascending=True)
        )
        primary_seed_rows = candidate_rows[candidate_rows["seed"] == int(primary_seed)][
            ["test_size", "RMSE_test", "R2_test"]
        ].rename(
            columns={
                "RMSE_test": "RMSE_test_primary_seed",
                "R2_test": "R2_test_primary_seed",
            }
        )
        block = block.merge(primary_seed_rows, on="test_size", how="left")
        ordered_cols = [
            "test_size",
            "n_seeds",
            "RMSE_test_primary_seed",
            "RMSE_test_mean",
            "RMSE_test_std",
            "RMSE_test_max",
            "R2_test_primary_seed",
            "R2_test_mean",
            "Spearmanr_test_mean",
        ]
        block = block[ordered_cols]
        block.insert(
            0,
            "primary_marker",
            ["P" if abs(float(value) - float(primary_test_size)) < 1.0e-12 else "" for value in block["test_size"]],
        )
        output.append((item, block.reset_index(drop=True)))
    return output


def ordered_summary(summary: pd.DataFrame, sort_blocks: str, top_blocks: int) -> pd.DataFrame:
    if sort_blocks == "model_descriptor":
        out = summary.sort_values(["model_name", "descript"], ascending=True)
    else:
        out = summary.sort_values(["RMSE_test_mean", "RMSE_test_std", "RMSE_test_max"], ascending=True)
    if top_blocks and top_blocks > 0:
        out = out.head(top_blocks)
    return out.reset_index(drop=True)


def sort_block_rows(block: pd.DataFrame, sort_rows: str) -> pd.DataFrame:
    if sort_rows == "rmse_test":
        return block.sort_values(["RMSE_test", "seed", "test_size"], ascending=True)
    return block.sort_values(["seed", "test_size"], ascending=True)


def with_local_marker(block: pd.DataFrame) -> pd.DataFrame:
    out = block.copy()
    best_rmse = float(out["RMSE_test"].min())
    out["marker"] = ["=>" if abs(float(value) - best_rmse) < 1.0e-12 else "" for value in out["RMSE_test"]]
    return out


def flat_rows_for_save(df: pd.DataFrame, summary: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for _, item in ordered_summary(summary, args.sort_blocks, args.top_blocks).iterrows():
        mask = (df["model_name"] == item["model_name"]) & (df["descript"] == item["descript"])
        block = with_local_marker(sort_block_rows(df[mask], args.sort_rows))
        block = block.assign(block_rank=int(item["rank"]))
        rows.append(block)
    if not rows:
        return pd.DataFrame()
    columns = [
        "block_rank",
        "marker",
        "target",
        "hyper_split",
        "search_method",
        "split",
        "model_name",
        "descript",
        "seed",
        "test_size",
        *METRIC_COLUMNS,
    ]
    return pd.concat(rows, ignore_index=True)[columns]


def color_enabled(mode: str) -> bool:
    if mode == "never":
        return False
    if mode == "always":
        return True
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def paint(text: str, style: str | None, enabled: bool) -> str:
    if not enabled or not style:
        return text
    return "\033[{}m{}\033[0m".format(style, text)


def format_for_terminal(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    float_cols = out.select_dtypes(include=["float"]).columns
    for col in float_cols:
        out[col] = out[col].map(lambda x: "" if pd.isna(x) else "{:.4f}".format(float(x)))
    int_cols = out.select_dtypes(include=["integer"]).columns
    for col in int_cols:
        out[col] = out[col].map(lambda x: "" if pd.isna(x) else str(int(x)))
    return out


def cell_style(column: str, value: object, local_best: bool) -> str | None:
    if local_best:
        return "4;1"
    if column == "model_name":
        return MODEL_STYLES.get(str(value), "1")
    if column == "descript":
        return "36"
    if column == "primary_marker" and str(value) == "P":
        return "1;35"
    if column in {"seed", "test_size", "split", "marker"}:
        return "35"
    return None


def print_table(df: pd.DataFrame, color_mode: str) -> None:
    printable = format_for_terminal(df)
    enabled = color_enabled(color_mode)
    columns = list(printable.columns)
    widths = {
        col: max(len(str(col)), *(len(str(value)) for value in printable[col].tolist()))
        for col in columns
    }
    text_cols = {"marker", "primary_marker", "model_name", "descript", "target", "hyper_split", "search_method", "split"}
    numeric_cols = {col for col in columns if col not in text_cols}

    header_cells = [
        paint(str(col).rjust(widths[col]) if col in numeric_cols else str(col).ljust(widths[col]), "1;37", enabled)
        for col in columns
    ]
    print(" ".join(header_cells))
    print(paint(" ".join("-" * widths[col] for col in columns), "2", enabled))

    for idx, row in printable.iterrows():
        raw_row = df.iloc[idx]
        local_best = str(raw_row.get("marker", "")) == "=>"
        cells = []
        for col in columns:
            text = str(row[col])
            padded = text.rjust(widths[col]) if col in numeric_cols else text.ljust(widths[col])
            cells.append(paint(padded, cell_style(col, raw_row[col], local_best), enabled))
        print(" ".join(cells))


def print_header(df: pd.DataFrame, summary: pd.DataFrame, input_fp: Path, color_mode: str) -> None:
    enabled = color_enabled(color_mode)
    title = "AAReact model metrics"
    print(paint(title, "1;36", enabled))
    print(paint("=" * len(title), "36", enabled))
    print("Input: {}".format(input_fp))
    print(
        "Context: target={} | hyper_split={} | search_method={}".format(
            ",".join(sorted(df["target"].astype(str).unique())),
            ",".join(sorted(df["hyper_split"].astype(str).unique())),
            ",".join(sorted(df["search_method"].astype(str).unique())),
        )
    )
    print("Rows: {} | Blocks: {}".format(len(df), len(summary)))
    print("Columns: RMSE/MAE lower is better; R2/Spearmanr higher is better.")
    print("Marker: => local best RMSE_test within each model+descriptor block.")
    best_idx = df["RMSE_test"].idxmin()
    best = df.loc[best_idx]
    print(
        "Global best by RMSE_test: => model={} descript={} seed={} test_size={:.4f} RMSE_test={:.4f} R2_test={:.4f}".format(
            best["model_name"],
            best["descript"],
            int(best["seed"]),
            float(best["test_size"]),
            float(best["RMSE_test"]),
            float(best["R2_test"]),
        )
    )
    print("")


def print_primary_header(df: pd.DataFrame, primary_summary: pd.DataFrame, input_fp: Path, args: argparse.Namespace) -> None:
    enabled = color_enabled(args.color)
    title = "AAReact model selection"
    print(paint(title, "1;36", enabled))
    print(paint("=" * len(title), "36", enabled))
    print("Input: {}".format(input_fp))
    print(
        "Context: target={} | hyper_split={} | search_method={}".format(
            ",".join(sorted(df["target"].astype(str).unique())),
            ",".join(sorted(df["hyper_split"].astype(str).unique())),
            ",".join(sorted(df["search_method"].astype(str).unique())),
        )
    )
    print(
        "Protocol: select model+descriptor using only seed={} and test_size={:.4f}; other seeds/test_size values are stability evidence only.".format(
            int(args.primary_seed),
            float(args.primary_test_size)
        )
    )
    print("Rows: {} | Candidates: {}".format(len(df), len(primary_summary)))
    print("Ranking: RMSE_test asc, then MAE_test asc, then R2_test desc.")
    print("Columns: RMSE/MAE lower is better; R2/Spearmanr higher is better.")
    print("Marker: P = primary test_size row; primary_seed columns show the selected seed within stability blocks.")
    print("")


def primary_rows_for_save(
    primary_display: pd.DataFrame,
    stability_rows: list[tuple[pd.Series, pd.DataFrame]],
) -> pd.DataFrame:
    saved_rows: list[pd.DataFrame] = []
    if not primary_display.empty:
        primary = primary_display.copy()
        primary.insert(0, "section", "primary_selection")
        saved_rows.append(primary)

    for item, block in stability_rows:
        stability = block.copy()
        stability.insert(0, "section", "test_size_stability")
        stability.insert(1, "rank", int(item["rank"]))
        stability.insert(2, "model_name", item["model_name"])
        stability.insert(3, "descript", item["descript"])
        saved_rows.append(stability)

    if not saved_rows:
        return pd.DataFrame()
    return pd.concat(saved_rows, ignore_index=True, sort=False)


def print_primary_report(df: pd.DataFrame, primary_summary: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    enabled = color_enabled(args.color)
    selected = primary_summary
    if args.top_blocks and args.top_blocks > 0:
        selected = selected.head(args.top_blocks)
    primary_display = selected[PRIMARY_COLUMNS].reset_index(drop=True)

    print(
        paint(
            "Primary selection: seed={} test_size={:.4f}".format(int(args.primary_seed), float(args.primary_test_size)),
            "1;37",
            enabled,
        )
    )
    print_table(primary_display, args.color)
    print("")

    print(paint("Stability across test_size", "1;37", enabled))
    stability_rows = build_stability_rows(df, primary_summary, args.primary_seed, args.primary_test_size, args.top_blocks)
    for item, block in stability_rows:
        model = str(item["model_name"])
        descriptor = str(item["descript"])
        title = "[#{:02d}] {} | {}".format(int(item["rank"]), model, descriptor)
        print(paint(title, MODEL_STYLES.get(model, "1"), enabled))
        print_table(block[STABILITY_COLUMNS].reset_index(drop=True), args.color)
        print("")

    return primary_rows_for_save(primary_display, stability_rows)


def print_blocks(df: pd.DataFrame, summary: pd.DataFrame, args: argparse.Namespace) -> None:
    enabled = color_enabled(args.color)
    columns = FULL_COLUMNS if args.full else KEY_COLUMNS
    blocks = ordered_summary(summary, args.sort_blocks, args.top_blocks)
    for position, item in blocks.iterrows():
        mask = (df["model_name"] == item["model_name"]) & (df["descript"] == item["descript"])
        block = with_local_marker(sort_block_rows(df[mask], args.sort_rows))
        best = block[block["marker"] == "=>"].iloc[0]
        model = str(item["model_name"])
        descriptor = str(item["descript"])
        block_title = (
            "[{:02d}] {} | {} | n={} | mean_RMSE_test={:.4f} | std={:.4f} | best: seed={} test_size={:.4f} RMSE_test={:.4f}".format(
                int(position) + 1,
                model,
                descriptor,
                int(item["n_splits"]),
                float(item["RMSE_test_mean"]),
                float(item["RMSE_test_std"]) if pd.notna(item["RMSE_test_std"]) else 0.0,
                int(best["seed"]),
                float(best["test_size"]),
                float(best["RMSE_test"]),
            )
        )
        print(paint(block_title, MODEL_STYLES.get(model, "1"), enabled))
        print_table(block[columns].reset_index(drop=True), args.color)
        print("")


def print_summary(summary: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    blocks = ordered_summary(summary, args.sort_blocks, args.top_blocks)
    output = blocks[SUMMARY_COLUMNS].reset_index(drop=True)
    print_table(output, args.color)
    return output


def print_flat(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.sort_values(["RMSE_test", "model_name", "descript", "seed", "test_size"], ascending=True)
    if args.top and args.top > 0:
        out = out.head(args.top)
    out = out.copy()
    out.insert(0, "marker", "")
    columns = ["marker", "model_name", "descript", "seed", "test_size", *METRIC_COLUMNS]
    if not args.full:
        columns = ["marker", "model_name", "descript", "seed", "test_size", "RMSE_test", "R2_test", "Spearmanr_test", "MAE_test", "RMSE_train", "R2_train"]
    output = out[columns].reset_index(drop=True)
    print_table(output, args.color)
    return output


def main() -> None:
    args = parse_args()
    df = apply_filters(load_metrics(args.input), args)
    summary = build_summary(df)

    if args.view == "primary":
        primary_summary = build_primary_summary(df, args.primary_seed, args.primary_test_size)
        print_primary_header(df, primary_summary, args.input, args)
        output_df = print_primary_report(df, primary_summary, args)
    elif args.view == "blocks":
        print_header(df, summary, args.input, args.color)
        print_blocks(df, summary, args)
        output_df = flat_rows_for_save(df, summary, args)
    elif args.view == "summary":
        print_header(df, summary, args.input, args.color)
        output_df = print_summary(summary, args)
    else:
        print_header(df, summary, args.input, args.color)
        output_df = print_flat(df, args)

    if args.save:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
