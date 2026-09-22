#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TRAIN_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = TRAIN_DIR / "train.csv"
DEFAULT_OUTPUT_DIR = TRAIN_DIR / "Analysis" / "train_metrics"
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
MODEL_ORDER = ["lgb", "xgb", "rf"]
MODEL_COLORS = {
    "lgb": "#4C78A8",
    "xgb": "#B55A30",
    "rf": "#2F8F83",
}
DESCRIPTOR_LINE_COLORS = [
    "#4C78A8",
    "#B55A30",
    "#2F8F83",
    "#7E6AAD",
    "#D6A13D",
    "#6C8EAD",
    "#8F6B4F",
    "#6E8E4E",
]
HEATMAP_CMAP = "YlGnBu"
AXIS_COLOR = "#222222"
GRID_COLOR = "#E7E7E7"
LIGHT_GRID_COLOR = "#F1F1F1"
LEGEND_EDGE_COLOR = "#B8B8B8"
ROW_BAND_COLOR = "#F7F7F7"
ROW_LINE_COLOR = "#ECECEC"
WHISKER_COLOR = "#5F5F5F"
NEUTRAL_MARKER_COLOR = "#777777"
PRIMARY_SELECTION_COLOR = "#D81B60"
DESCRIPTOR_FAMILY_MARKERS = {
    "rdkit": "o",
    "xtb": "s",
    "geometry": "^",
    "rdkit+geometry": "D",
    "xtb+geometry": "v",
    "rdkit+xtb": "P",
    "all": "h",
}
DESCRIPTOR_FAMILY_LABELS = {
    "rdkit": "RDKit only (2D)",
    "xtb": "xTB only (quantum)",
    "geometry": "SOAP/ACSF only (geometry)",
    "rdkit+geometry": "RDKit + SOAP/ACSF",
    "xtb+geometry": "xTB + SOAP/ACSF",
    "rdkit+xtb": "RDKit + xTB",
    "all": "RDKit + xTB + SOAP/ACSF",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize AAReact ML train metrics from train.csv.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--top-box",
        type=int,
        default=0,
        help="Number of descriptors shown in the boxplot; 0 shows all descriptors.",
    )
    parser.add_argument("--top-lines", type=int, default=6, help="Number of descriptors shown in sensitivity plots.")
    parser.add_argument("--primary-seed", type=int, default=PRIMARY_SEED, help="Seed used for primary model selection.")
    parser.add_argument(
        "--primary-test-size",
        type=float,
        default=PRIMARY_TEST_SIZE,
        help="Test size used for primary model selection.",
    )
    return parser.parse_args()


def split_sort_key(split_name: str) -> tuple[int, float]:
    parts = str(split_name).split("_")
    if len(parts) >= 4 and parts[0] == "seed" and parts[2] == "test":
        return int(parts[1]), float(parts[3].replace("-", "."))
    return 999999, 999999.0


def prepare_data(input_fp: Path) -> pd.DataFrame:
    if not input_fp.exists():
        raise FileNotFoundError("Input CSV does not exist: {}".format(input_fp))

    df = pd.read_csv(input_fp)
    required = {"target", "hyper_split", "split", "model", "descriptor", *METRIC_COLUMNS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError("{} is missing required columns: {}".format(input_fp, ", ".join(missing)))

    if "search_method" not in df.columns:
        df["search_method"] = "legacy"
    df["search_method"] = df["search_method"].fillna("legacy").astype(str)

    if "seed" not in df.columns or "test_size" not in df.columns:
        parsed = df["split"].map(split_sort_key)
        df["seed"] = [item[0] for item in parsed]
        df["test_size"] = [item[1] for item in parsed]

    for col in METRIC_COLUMNS + ["seed", "test_size"]:
        df[col] = pd.to_numeric(df[col], errors="raise")

    df["model_descriptor"] = df["model"].astype(str) + "__" + df["descriptor"].astype(str)
    df["RMSE_gap"] = df["RMSE_test"] - df["RMSE_train"]
    df["R2_gap"] = df["R2_train"] - df["R2_test"]
    df["split"] = pd.Categorical(
        df["split"],
        categories=sorted(df["split"].unique(), key=split_sort_key),
        ordered=True,
    )
    return df


def is_close(series: pd.Series, value: float) -> pd.Series:
    return (series.astype(float) - float(value)).abs() < 1.0e-12


def descriptor_family(descriptor: str) -> str:
    parts = set(str(descriptor).split("_"))
    has_rdkit = "rdkit" in parts
    has_xtb = "xtb" in parts
    has_geometry = bool(parts & {"soap", "acsf"})

    if has_rdkit and has_xtb and has_geometry:
        return "all"
    if has_rdkit and has_xtb:
        return "rdkit+xtb"
    if has_rdkit and has_geometry:
        return "rdkit+geometry"
    if has_xtb and has_geometry:
        return "xtb+geometry"
    if has_rdkit:
        return "rdkit"
    if has_xtb:
        return "xtb"
    if has_geometry:
        return "geometry"
    return "other"


def format_descriptor_label(descriptor: str) -> str:
    label_map = {
        "rdkit": "RDKit",
        "soap": "SOAP",
        "acsf": "ACSF",
        "xtb": "xTB",
    }
    return "+".join(label_map.get(part, part.upper()) for part in str(descriptor).split("_"))


def format_split_label(split_name: str) -> str:
    seed, test_size = split_sort_key(split_name)
    if seed == 999999:
        return str(split_name)
    return "s{} / t{:.2f}".format(seed, test_size)


def format_heatmap_label(label: object) -> str:
    text = str(label)
    if text in MODEL_ORDER:
        return text.upper()
    if "__" in text:
        model, descriptor = text.split("__", 1)
        return "{}@{}".format(model.upper(), format_descriptor_label(descriptor))
    if text.startswith("seed_"):
        return format_split_label(text)
    return format_descriptor_label(text)


def apply_publication_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "legend.title_fontsize": 7,
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def save_summary(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    summary = (
        df.groupby(["target", "hyper_split", "search_method", "model", "descriptor"], observed=True)
        .agg(
            n=("RMSE_test", "size"),
            RMSE_test_mean=("RMSE_test", "mean"),
            RMSE_test_std=("RMSE_test", "std"),
            RMSE_test_min=("RMSE_test", "min"),
            RMSE_test_max=("RMSE_test", "max"),
            MAE_test_mean=("MAE_test", "mean"),
            MAE_test_std=("MAE_test", "std"),
            R2_test_mean=("R2_test", "mean"),
            R2_test_std=("R2_test", "std"),
            Spearmanr_test_mean=("Spearmanr_test", "mean"),
            Spearmanr_test_std=("Spearmanr_test", "std"),
            RMSE_train_mean=("RMSE_train", "mean"),
            RMSE_gap_mean=("RMSE_gap", "mean"),
            RMSE_gap_std=("RMSE_gap", "std"),
            R2_gap_mean=("R2_gap", "mean"),
            R2_gap_std=("R2_gap", "std"),
        )
        .reset_index()
        .sort_values(["RMSE_test_mean", "RMSE_test_std", "RMSE_gap_mean"])
    )
    summary.insert(0, "rank_by_RMSE_test", np.arange(1, len(summary) + 1))
    summary.insert(
        summary.columns.get_loc("descriptor") + 1,
        "descriptor_family",
        summary["descriptor"].map(descriptor_family),
    )
    summary.to_csv(output_dir / "summary_by_model_descriptor.csv", index=False)
    return summary


def save_primary_selection(
    df: pd.DataFrame,
    output_dir: Path,
    primary_seed: int,
    primary_test_size: float,
) -> pd.DataFrame:
    primary = df[(df["seed"] == int(primary_seed)) & is_close(df["test_size"], primary_test_size)].copy()
    if primary.empty:
        available = (
            df[["seed", "test_size"]]
            .drop_duplicates()
            .sort_values(["seed", "test_size"])
        )
        split_labels = [
            "seed={} test_size={:.4f}".format(int(row.seed), float(row.test_size))
            for row in available.itertuples(index=False)
        ]
        raise ValueError(
            "No rows found for primary seed={} and test_size={:.4f}. Available splits: {}".format(
                int(primary_seed),
                float(primary_test_size),
                ", ".join(split_labels),
            )
        )

    columns = [
        "target",
        "hyper_split",
        "search_method",
        "split",
        "seed",
        "test_size",
        "model",
        "descriptor",
        "RMSE_test",
        "MAE_test",
        "R2_test",
        "Spearmanr_test",
        "RMSE_train",
        "RMSE_gap",
    ]
    primary = primary[columns].sort_values(
        ["RMSE_test", "MAE_test", "R2_test"],
        ascending=[True, True, False],
    )
    primary.insert(0, "primary_rank", np.arange(1, len(primary) + 1))
    primary.insert(
        primary.columns.get_loc("descriptor") + 1,
        "descriptor_family",
        primary["descriptor"].map(descriptor_family),
    )
    primary.to_csv(output_dir / "primary_selection_by_model_descriptor.csv", index=False)
    return primary


def ordered_models(df: pd.DataFrame) -> list[str]:
    present = set(df["model"].unique())
    return [model for model in MODEL_ORDER if model in present] + sorted(present - set(MODEL_ORDER))


def ordered_descriptors(summary: pd.DataFrame, primary_selection: pd.DataFrame | None = None) -> list[str]:
    if primary_selection is not None and not primary_selection.empty:
        ordered: list[str] = []
        seen: set[str] = set()
        for descriptor in primary_selection["descriptor"].astype(str):
            if descriptor not in seen:
                ordered.append(descriptor)
                seen.add(descriptor)
        for descriptor in ordered_descriptors(summary):
            if descriptor not in seen:
                ordered.append(descriptor)
                seen.add(descriptor)
        return ordered

    by_desc = (
        summary.groupby("descriptor", observed=True)["RMSE_test_mean"]
        .mean()
        .sort_values()
    )
    return by_desc.index.tolist()


def plot_heatmap(
    matrix: pd.DataFrame,
    title: str,
    output_fp: Path,
    cmap: str,
    fmt: str,
    cbar_label: str,
) -> None:
    row_scale = 0.22 if len(matrix.index) > 25 else 0.31
    text_size = 4.8 if len(matrix.index) > 25 else 6.5
    ytick_size = 5.6 if len(matrix.index) > 25 else 8
    fig_h = max(4.2, row_scale * len(matrix.index) + 1.70)
    fig_w = max(4.8, 0.64 * len(matrix.columns) + 2.45)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    values = matrix.to_numpy(dtype=float)
    image = ax.imshow(values, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_xticklabels([format_heatmap_label(col) for col in matrix.columns])
    ax.set_yticklabels([format_heatmap_label(idx) for idx in matrix.index])
    ax.tick_params(axis="y", labelsize=ytick_size)
    ax.set_title(title, pad=7)
    rotation = 42 if len(matrix.columns) > 4 else 0
    ax.tick_params(axis="x", rotation=rotation)
    if rotation:
        for label in ax.get_xticklabels():
            label.set_ha("right")
            label.set_rotation_mode("anchor")

    ax.set_xticks(np.arange(values.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(values.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", direction="out", top=False, right=False, width=0.8)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(AXIS_COLOR)
        spine.set_linewidth(0.8)

    if all(str(col) in MODEL_ORDER for col in matrix.columns):
        ax.set_xlabel("Model")
    elif all(str(col).startswith("seed_") for col in matrix.columns):
        ax.set_xlabel("Split setting")
    if any("__" in str(idx) for idx in matrix.index):
        ax.set_ylabel("Model@descriptor")
    else:
        ax.set_ylabel("Descriptor combination")

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if np.isfinite(value):
                rgba = image.cmap(image.norm(value))
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                text_color = "white" if luminance < 0.48 else AXIS_COLOR
                ax.text(
                    j,
                    i,
                    format(value, fmt),
                    ha="center",
                    va="center",
                    fontsize=text_size,
                    color=text_color,
                )

    cbar = fig.colorbar(image, ax=ax, shrink=0.82, pad=0.018)
    cbar.set_label(cbar_label)
    cbar.outline.set_linewidth(0.6)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(output_fp.with_suffix(".{}".format(suffix)), bbox_inches="tight")
    plt.close(fig)


def plot_top_descriptor_boxplot(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    primary_selection: pd.DataFrame,
    output_dir: Path,
    top_n: int,
    primary_seed: int,
    primary_test_size: float,
) -> None:
    top_descriptors = ordered_descriptors(summary, primary_selection)
    if top_n > 0:
        top_descriptors = top_descriptors[:top_n]
    models = ordered_models(df)
    box_height = 0.18
    base = np.arange(len(top_descriptors), dtype=float)
    if len(models) == 1:
        offsets = np.array([0.0])
    else:
        offsets = np.linspace(-0.24, 0.24, len(models))

    fig, ax = plt.subplots(figsize=(6.85, max(4.9, 0.30 * len(top_descriptors) + 1.70)), dpi=300)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    for desc_idx in range(len(top_descriptors)):
        if desc_idx % 2 == 0:
            ax.axhspan(desc_idx - 0.45, desc_idx + 0.45, color=ROW_BAND_COLOR, zorder=0)
    for desc_idx in range(len(top_descriptors) - 1):
        ax.axhline(desc_idx + 0.5, color=ROW_LINE_COLOR, linewidth=0.45, zorder=1)

    for model_idx, model in enumerate(models):
        positions = base + offsets[model_idx]
        series = [
            df[(df["descriptor"] == desc) & (df["model"] == model)]["RMSE_test"].to_numpy()
            for desc in top_descriptors
        ]
        box = ax.boxplot(
            series,
            positions=positions,
            widths=box_height,
            vert=False,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
            medianprops={"color": AXIS_COLOR, "linewidth": 1.0},
            whiskerprops={"color": WHISKER_COLOR, "linewidth": 0.75},
            capprops={"color": WHISKER_COLOR, "linewidth": 0.75},
        )
        for patch in box["boxes"]:
            patch.set_facecolor(MODEL_COLORS.get(model, NEUTRAL_MARKER_COLOR))
            patch.set_alpha(0.45)
            patch.set_edgecolor(AXIS_COLOR)
            patch.set_linewidth(0.65)

        color = MODEL_COLORS.get(model, NEUTRAL_MARKER_COLOR)
        for desc_idx, values in enumerate(series):
            if len(values) == 0:
                continue
            jitter = np.linspace(-box_height * 0.28, box_height * 0.28, len(values))
            ax.scatter(
                values,
                np.full(len(values), positions[desc_idx]) + jitter,
                s=10,
                marker="o",
                facecolor=color,
                edgecolor="white",
                linewidth=0.25,
                alpha=0.78,
                zorder=3,
            )

    targets = set(df["target"].astype(str).str.lower())
    unit = r" (kcal mol$^{-1}$)" if targets == {"ddg"} else ""
    ax.set_yticks(base)
    ax.set_yticklabels([format_descriptor_label(desc) for desc in top_descriptors])
    ax.invert_yaxis()
    ax.set_xlabel("Test RMSE{}".format(unit))
    ax.set_ylabel("Descriptor combination")
    best = primary_selection.iloc[0]
    ax.set_title(
        "Primary order: {}@{} (seed {} / test {:.2f})".format(
            str(best["model"]).upper(),
            format_descriptor_label(best["descriptor"]),
            int(primary_seed),
            float(primary_test_size),
        ),
        pad=5,
    )
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.45)
    top_df = df[df["descriptor"].isin(top_descriptors)]
    x_min = float(top_df["RMSE_test"].min())
    x_max = float(top_df["RMSE_test"].max())
    x_range = max(x_max - x_min, 1e-6)
    ax.set_xlim(x_min - 0.025 * x_range, x_max + 0.08 * x_range)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(AXIS_COLOR)
        spine.set_linewidth(0.8)
    ax.tick_params(axis="both", direction="out", top=False, right=False, width=0.8)

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markersize=5.5,
            markerfacecolor=MODEL_COLORS.get(model, NEUTRAL_MARKER_COLOR),
            markeredgecolor=AXIS_COLOR,
            markeredgewidth=0.35,
            alpha=0.82,
            label=model.upper(),
        )
        for model in models
    ]
    legend = ax.legend(
        handles=handles,
        title="Model",
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        ncol=1,
        frameon=True,
        framealpha=0.94,
        facecolor="white",
        edgecolor=LEGEND_EDGE_COLOR,
        borderpad=0.18,
        handlelength=0.72,
        handletextpad=0.28,
        labelspacing=0.22,
    )
    legend.get_frame().set_linewidth(0.55)

    fig.tight_layout(pad=0.35)
    for suffix in ("png", "svg"):
        fig.savefig(output_dir / "rmse_test_box_top_descriptors.{}".format(suffix), bbox_inches="tight")
    plt.close(fig)


def plot_gap_scatter(
    df: pd.DataFrame,
    primary_selection: pd.DataFrame,
    output_dir: Path,
    primary_test_size: float,
) -> None:
    plot_df = df[is_close(df["test_size"], primary_test_size)].copy()
    if plot_df.empty:
        raise ValueError("No rows found for gap plot at test_size={:.4f}".format(float(primary_test_size)))

    plot_df["descriptor_family"] = plot_df["descriptor"].map(descriptor_family)
    plot_df = plot_df.sort_values(["seed", "RMSE_test", "RMSE_gap"])
    seeds = sorted(plot_df["seed"].astype(int).unique().tolist())
    models = ordered_models(plot_df)

    fig, axes = plt.subplots(
        1,
        len(seeds),
        figsize=(max(7.8, 3.0 * len(seeds)), 3.95),
        dpi=300,
        sharex=True,
        sharey=True,
    )
    if len(seeds) == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")

    targets = set(plot_df["target"].astype(str).str.lower())
    unit = r" (kcal mol$^{-1}$)" if targets == {"ddg"} else ""

    x_range = max(float(plot_df["RMSE_test"].max() - plot_df["RMSE_test"].min()), 1.0e-6)
    y_range = max(float(plot_df["RMSE_gap"].max() - plot_df["RMSE_gap"].min()), 1.0e-6)
    x_pad = 0.07 * x_range
    y_pad = 0.10 * y_range
    x_lim = (float(plot_df["RMSE_test"].min()) - x_pad, float(plot_df["RMSE_test"].max()) + x_pad)
    y_lim = (
        min(0.0, float(plot_df["RMSE_gap"].min()) - y_pad),
        float(plot_df["RMSE_gap"].max()) + y_pad,
    )

    selected = primary_selection.iloc[0]
    selected_model = str(selected["model"])
    selected_descriptor = str(selected["descriptor"])
    selected_label = "{}@{}".format(selected_model.upper(), format_descriptor_label(selected_descriptor))

    for ax, seed in zip(axes, seeds):
        ax.set_facecolor("white")
        seed_df = plot_df[plot_df["seed"].astype(int) == int(seed)]
        for family, marker in DESCRIPTOR_FAMILY_MARKERS.items():
            family_df = seed_df[seed_df["descriptor_family"] == family]
            if family_df.empty:
                continue
            for model in models:
                sub = family_df[family_df["model"] == model]
                if sub.empty:
                    continue
                color = MODEL_COLORS.get(model, NEUTRAL_MARKER_COLOR)
                size = 56 if family != "all" else 72
                ax.scatter(
                    sub["RMSE_test"],
                    sub["RMSE_gap"],
                    s=size,
                    marker=marker,
                    facecolor=color,
                    edgecolor=AXIS_COLOR,
                    linewidth=0.35,
                    alpha=0.84,
                    zorder=3,
                )

        highlight = seed_df[
            (seed_df["model"].astype(str) == selected_model)
            & (seed_df["descriptor"].astype(str) == selected_descriptor)
        ]
        if not highlight.empty:
            row = highlight.iloc[0]
            ax.scatter(
                [row["RMSE_test"]],
                [row["RMSE_gap"]],
                s=156,
                marker=DESCRIPTOR_FAMILY_MARKERS.get(descriptor_family(selected_descriptor), "o"),
                facecolor="none",
                edgecolor=PRIMARY_SELECTION_COLOR,
                linewidth=1.75,
                zorder=5,
            )

        ax.axhline(0.0, color=LEGEND_EDGE_COLOR, linewidth=0.55, linestyle="--", zorder=1)
        ax.set_title("seed = {}".format(int(seed)), pad=6)
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.grid(axis="both", color=GRID_COLOR, linewidth=0.42)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(AXIS_COLOR)
            spine.set_linewidth(0.8)
        ax.tick_params(axis="both", direction="out", top=False, right=False, width=0.8)

    axes[0].set_ylabel("RMSE gap (test - train){}".format(unit))
    for ax in axes:
        ax.set_xlabel("Test RMSE{}".format(unit))

    model_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=5.4,
            markerfacecolor=MODEL_COLORS.get(model, NEUTRAL_MARKER_COLOR),
            markeredgecolor=AXIS_COLOR,
            markeredgewidth=0.35,
            label=model.upper(),
        )
        for model in models
    ]
    family_order = [family for family in DESCRIPTOR_FAMILY_MARKERS if family in set(plot_df["descriptor_family"])]
    family_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=DESCRIPTOR_FAMILY_MARKERS[family],
            linestyle="none",
            markersize=5.1,
            markerfacecolor=NEUTRAL_MARKER_COLOR,
            markeredgecolor=AXIS_COLOR,
            markeredgewidth=0.35,
            label=DESCRIPTOR_FAMILY_LABELS.get(family, family),
        )
        for family in family_order
    ]
    selected_handle = plt.Line2D(
        [0],
        [0],
        marker=DESCRIPTOR_FAMILY_MARKERS.get(descriptor_family(selected_descriptor), "o"),
        linestyle="none",
        markersize=6.6,
        markerfacecolor="none",
        markeredgecolor=PRIMARY_SELECTION_COLOR,
        markeredgewidth=1.45,
        label="Primary selection",
    )

    fig.suptitle(
        "RMSE gap at test_size = {:.2f}; primary = {}".format(float(primary_test_size), selected_label),
        y=0.992,
        fontsize=9,
    )
    leg1 = fig.legend(
        handles=[*model_handles, selected_handle],
        title="Model",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.938),
        ncol=len(model_handles) + 1,
        frameon=True,
        framealpha=0.96,
        facecolor="white",
        edgecolor=LEGEND_EDGE_COLOR,
        borderpad=0.24,
        handletextpad=0.35,
        columnspacing=0.85,
    )
    leg1.get_frame().set_linewidth(0.55)
    fig.legend(
        handles=family_handles,
        title="Descriptor family",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=4,
        frameon=True,
        framealpha=0.96,
        facecolor="white",
        edgecolor=LEGEND_EDGE_COLOR,
        borderpad=0.22,
        handletextpad=0.30,
        columnspacing=0.75,
        labelspacing=0.35,
        fontsize=6.5,
    ).get_frame().set_linewidth(0.55)

    fig.tight_layout(rect=(0.0, 0.16, 1.0, 0.86), w_pad=1.0)
    for suffix in ("png", "svg"):
        fig.savefig(output_dir / "rmse_gap_vs_rmse_test.{}".format(suffix), bbox_inches="tight")
    plt.close(fig)


def plot_test_size_sensitivity(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    primary_selection: pd.DataFrame,
    output_dir: Path,
    top_n: int,
    primary_seed: int,
    primary_test_size: float,
) -> None:
    _ = (summary, top_n)
    models = ordered_models(df)
    selected = primary_selection.iloc[0]
    selected_model = str(selected["model"])
    selected_descriptor = str(selected["descriptor"])
    sensitivity_df = df[df["descriptor"].astype(str) == selected_descriptor].copy()
    if sensitivity_df.empty:
        raise ValueError("No rows found for sensitivity descriptor: {}".format(selected_descriptor))

    fig, ax = plt.subplots(figsize=(5.85, 3.35), dpi=300)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    x_values = sorted(sensitivity_df["test_size"].astype(float).unique())
    seeds = sorted(sensitivity_df["seed"].astype(int).unique())
    seed_marker_cycle = ["o", "D", "^", "s", "v", "P"]
    seed_markers = {seed: seed_marker_cycle[idx % len(seed_marker_cycle)] for idx, seed in enumerate(seeds)}
    model_gap = 0.72
    group_gap = 0.86
    mean_offset = 0.18
    seed_offsets = {
        seed: offset
        for seed, offset in zip(seeds, np.linspace(-0.21, -0.07, len(seeds)))
    }
    slot_positions: dict[tuple[float, str], float] = {}
    group_centers: dict[float, float] = {}
    group_spans: dict[float, tuple[float, float]] = {}
    group_width = (len(models) - 1) * model_gap
    for test_idx, test_size in enumerate(x_values):
        group_start = test_idx * (group_width + group_gap)
        group_positions = []
        for model_idx, model in enumerate(models):
            position = group_start + model_idx * model_gap
            slot_positions[(float(test_size), model)] = position
            group_positions.append(position)
        group_centers[float(test_size)] = float(np.mean(group_positions))

    group_keys = [float(test_size) for test_size in x_values]
    separator_positions = [
        0.5 * (group_centers[left_test] + group_centers[right_test])
        for left_test, right_test in zip(group_keys[:-1], group_keys[1:])
    ]
    if separator_positions:
        first_half_width = separator_positions[0] - group_centers[group_keys[0]]
        last_half_width = group_centers[group_keys[-1]] - separator_positions[-1]
    else:
        first_half_width = 0.5 * group_width + 0.45
        last_half_width = first_half_width
    for test_idx, test_size in enumerate(group_keys):
        span_left = separator_positions[test_idx - 1] if test_idx > 0 else group_centers[test_size] - first_half_width
        span_right = (
            separator_positions[test_idx]
            if test_idx < len(group_keys) - 1
            else group_centers[test_size] + last_half_width
        )
        group_spans[test_size] = (span_left, span_right)
    y_low = np.inf
    y_high = -np.inf

    for test_idx, test_size in enumerate(x_values):
        test_size = float(test_size)
        span_left, span_right = group_spans[test_size]
        if np.isclose(test_size, float(primary_test_size)):
            facecolor = "#FFF2F6"
            alpha = 0.72
        else:
            facecolor = "#F7F9FB" if test_idx % 2 == 0 else "#FFFFFF"
            alpha = 0.80
        ax.axvspan(
            span_left,
            span_right,
            facecolor=facecolor,
            edgecolor="none",
            linewidth=0.0,
            alpha=alpha,
            zorder=0,
        )

    for separator_x in separator_positions:
        ax.axvline(
            separator_x,
            color="#D9DEE4",
            linewidth=0.75,
            zorder=1,
        )

    for model in models:
        sub = sensitivity_df[sensitivity_df["model"].astype(str) == model].copy()
        if sub.empty:
            continue
        color = MODEL_COLORS.get(model, NEUTRAL_MARKER_COLOR)
        grouped = (
            sub.groupby("test_size", observed=True)
            .agg(mean=("RMSE_test", "mean"), std=("RMSE_test", "std"))
            .reset_index()
            .sort_values("test_size")
        )
        std = grouped["std"].fillna(0.0)
        y_low = min(y_low, float(min(sub["RMSE_test"].min(), (grouped["mean"] - std).min())))
        y_high = max(y_high, float(max(sub["RMSE_test"].max(), (grouped["mean"] + std).max())))
        for _, row in grouped.iterrows():
            test_size = float(row["test_size"])
            base_x = slot_positions[(test_size, model)]
            ax.errorbar(
                base_x + mean_offset,
                float(row["mean"]),
                yerr=float(0.0 if pd.isna(row["std"]) else row["std"]),
                fmt="o",
                linestyle="none",
                color=color,
                markersize=5.0,
                elinewidth=0.95,
                capsize=2.5,
                capthick=0.85,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.4,
                alpha=0.95,
                zorder=4 if model == selected_model else 3,
            )
        for _, row in sub.iterrows():
            seed = int(row["seed"])
            test_size = float(row["test_size"])
            base_x = slot_positions[(test_size, model)]
            ax.scatter(
                base_x + seed_offsets.get(seed, -0.14),
                float(row["RMSE_test"]),
                s=17,
                marker=seed_markers.get(seed, "o"),
                facecolor="white",
                edgecolor=color,
                linewidth=0.62,
                alpha=0.68,
                zorder=5,
            )

    selected_row = sensitivity_df[
        (sensitivity_df["model"].astype(str) == selected_model)
        & (sensitivity_df["seed"].astype(int) == int(primary_seed))
        & is_close(sensitivity_df["test_size"], primary_test_size)
    ]
    if not selected_row.empty:
        row = selected_row.iloc[0]
        selected_y = float(row["RMSE_test"])
        y_low = min(y_low, selected_y)
        y_high = max(y_high, selected_y)
        ax.text(
            0.98,
            0.96,
            "Primary: {} seed {}, test={:.2f}".format(
                selected_model.upper(),
                int(primary_seed),
                float(primary_test_size),
            ),
            transform=ax.transAxes,
            ha="right",
            va="top",
            color=PRIMARY_SELECTION_COLOR,
            fontsize=5.9,
            zorder=7,
        )

    targets = set(df["target"].astype(str).str.lower())
    unit = r" (kcal mol$^{-1}$)" if targets == {"ddg"} else ""
    if np.isfinite(y_low) and np.isfinite(y_high):
        y_range = max(y_high - y_low, 1.0e-6)
        ax.set_ylim(max(0.0, y_low - 0.12 * y_range), y_high + 0.18 * y_range)
    all_slot_positions = [slot_positions[(float(test_size), model)] for test_size in x_values for model in models]
    ax.set_xlim(group_spans[float(x_values[0])][0], group_spans[float(x_values[-1])][1])
    ax.set_xticks(all_slot_positions)
    ax.set_xticklabels([model.upper() for _ in x_values for model in models], fontsize=6.8)
    for test_size in x_values:
        ax.text(
            group_centers[float(test_size)],
            -0.16,
            "test={:.2f}".format(float(test_size)),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=6.8,
            color=NEUTRAL_MARKER_COLOR,
            clip_on=False,
        )
    ax.set_xlabel("")
    ax.set_ylabel("Test RMSE{}".format(unit))
    ax.set_title("Test-size sensitivity for {}".format(format_descriptor_label(selected_descriptor)), pad=7)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.45)
    ax.grid(axis="x", visible=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(AXIS_COLOR)
        spine.set_linewidth(0.8)
    ax.tick_params(axis="both", direction="out", top=False, right=False, width=0.8)

    model_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=4.4,
            color=MODEL_COLORS.get(model, NEUTRAL_MARKER_COLOR),
            markerfacecolor=MODEL_COLORS.get(model, NEUTRAL_MARKER_COLOR),
            markeredgecolor="white",
            markeredgewidth=0.35,
            label=model.upper(),
        )
        for model in models
    ]
    seed_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=seed_markers.get(seed, "o"),
            linestyle="none",
            markersize=4.6,
            markerfacecolor="white",
            markeredgecolor=NEUTRAL_MARKER_COLOR,
            markeredgewidth=0.75,
            label="seed {}".format(seed),
        )
        for seed in seeds
    ]
    model_legend = ax.legend(
        handles=model_handles,
        title="Mean +/- SD",
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=True,
        framealpha=0.94,
        facecolor="white",
        edgecolor=LEGEND_EDGE_COLOR,
        borderpad=0.25,
        handlelength=1.25,
        handletextpad=0.45,
        labelspacing=0.35,
    )
    model_legend.get_frame().set_linewidth(0.55)
    ax.add_artist(model_legend)
    seed_legend = ax.legend(
        handles=seed_handles,
        title="Open points",
        loc="lower right",
        bbox_to_anchor=(0.98, 0.04),
        frameon=True,
        framealpha=0.94,
        facecolor="white",
        edgecolor=LEGEND_EDGE_COLOR,
        borderpad=0.25,
        handlelength=0.85,
        handletextpad=0.45,
        labelspacing=0.35,
    )
    seed_legend.get_frame().set_linewidth(0.55)

    fig.tight_layout(pad=0.35)
    for suffix in ("png", "svg"):
        fig.savefig(output_dir / "test_size_sensitivity.{}".format(suffix), bbox_inches="tight")
    plt.close(fig)


def write_manifest(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    outputs: list[str],
    args: argparse.Namespace,
) -> None:
    manifest = {
        "source_rows": int(len(df)),
        "summary_rows": int(len(summary)),
        "primary_seed": int(args.primary_seed),
        "primary_test_size": float(args.primary_test_size),
        "rmse_gap_vs_rmse_test_scope": "test_size={:.4f}; panels are seeds".format(float(args.primary_test_size)),
        "targets": sorted(df["target"].astype(str).unique().tolist()),
        "hyper_splits": sorted(df["hyper_split"].astype(str).unique().tolist()),
        "search_methods": sorted(df["search_method"].astype(str).unique().tolist()),
        "train_splits": sorted(df["split"].astype(str).unique().tolist(), key=split_sort_key),
        "outputs": outputs,
    }
    (output_dir / "plot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def remove_stale_outputs(output_dir: Path) -> None:
    for stem in ("rmse_test_mean_heatmap", "rmse_test_std_heatmap", "split_heatmap_top_models"):
        for suffix in ("png", "svg"):
            fp = output_dir / "{}.{}".format(stem, suffix)
            if fp.exists():
                fp.unlink()


def render_plots(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    primary_selection: pd.DataFrame,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    outputs = ["summary_by_model_descriptor.csv", "primary_selection_by_model_descriptor.csv"]
    remove_stale_outputs(output_dir)
    plot_top_descriptor_boxplot(
        df,
        summary,
        primary_selection,
        output_dir,
        args.top_box,
        args.primary_seed,
        args.primary_test_size,
    )
    outputs.extend([
        "rmse_test_box_top_descriptors.png",
        "rmse_test_box_top_descriptors.svg",
    ])
    plot_gap_scatter(df, primary_selection, output_dir, args.primary_test_size)
    outputs.extend([
        "rmse_gap_vs_rmse_test.png",
        "rmse_gap_vs_rmse_test.svg",
    ])
    plot_test_size_sensitivity(
        df,
        summary,
        primary_selection,
        output_dir,
        args.top_lines,
        args.primary_seed,
        args.primary_test_size,
    )
    outputs.extend([
        "test_size_sensitivity.png",
        "test_size_sensitivity.svg",
    ])
    return outputs


def main() -> None:
    args = parse_args()
    apply_publication_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    remove_stale_outputs(args.output_dir)
    df = prepare_data(args.input)
    summary = save_summary(df, args.output_dir)
    primary_selection = save_primary_selection(df, args.output_dir, args.primary_seed, args.primary_test_size)

    search_methods = sorted(df["search_method"].astype(str).unique().tolist())
    if len(search_methods) <= 1:
        outputs = render_plots(df, summary, primary_selection, args.output_dir, args)
        write_manifest(df, summary, args.output_dir, outputs, args)
        outputs.append("plot_manifest.json")
    else:
        outputs = ["summary_by_model_descriptor.csv", "primary_selection_by_model_descriptor.csv"]
        for search_method in search_methods:
            sub_output_dir = args.output_dir / "search_{}".format(search_method)
            sub_output_dir.mkdir(parents=True, exist_ok=True)
            sub_df = df[df["search_method"] == search_method].copy()
            sub_summary = save_summary(sub_df, sub_output_dir)
            sub_primary_selection = save_primary_selection(
                sub_df,
                sub_output_dir,
                args.primary_seed,
                args.primary_test_size,
            )
            sub_args = argparse.Namespace(**vars(args))
            sub_args.output_dir = sub_output_dir
            sub_outputs = render_plots(sub_df, sub_summary, sub_primary_selection, sub_output_dir, sub_args)
            write_manifest(sub_df, sub_summary, sub_output_dir, sub_outputs, sub_args)
            outputs.append("search_{}/".format(search_method))
        write_manifest(df, summary, args.output_dir, outputs, args)
        outputs.append("plot_manifest.json")

    print(json.dumps({
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "rows": int(len(df)),
        "summary_rows": int(len(summary)),
        "primary_seed": int(args.primary_seed),
        "primary_test_size": float(args.primary_test_size),
        "primary_best": {
            "model": str(primary_selection.iloc[0]["model"]),
            "descriptor": str(primary_selection.iloc[0]["descriptor"]),
            "RMSE_test": float(primary_selection.iloc[0]["RMSE_test"]),
        },
        "outputs": outputs,
    }, indent=2))


if __name__ == "__main__":
    main()
