#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import re
import tomllib
from pathlib import Path

TRAIN_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAIN_DIR.parent
os.environ.setdefault("MPLCONFIGDIR", str(TRAIN_DIR / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(TRAIN_DIR / ".cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


DEFAULT_TARGET = "ddg"
DEFAULT_HYPER_SPLIT = "seed_1_test_0-2"
DEFAULT_SEARCH_METHOD = "optuna"
DEFAULT_SPLIT = "seed_1_test_0-2"
DEFAULT_MODEL_NAME = "xgb"
DEFAULT_DESCRIPTOR = "rdkit_xtb_acsf"
DEFAULT_OUTPUT_DIR = TRAIN_DIR / "Analysis" / "selected_model"
DDG_R_KCAL = 0.001987
CELSIUS_TO_KELVIN = 273.15

MODEL_COLORS = {
    "train": "#4C78A8",
    "test": "#B55A30",
}
SOURCE_COLORS = {
    "rdkit": "#4C78A8",
    "xtb": "#B55A30",
    "acsf": "#2F8F83",
    "soap": "#7E6AAD",
    "condition": "#7E6AAD",
    "other": "#777777",
}
ROLE_COLORS = {
    "catalyst": "#0072B2",
    "reactant": "#D55E00",
    "solvent": "#009E73",
    "pressure": "#CC79A7",
    "temperature": "#999999",
    "other": "#777777",
}
PRIMARY_SELECTION_COLOR = "#D81B60"
AXIS_COLOR = "#222222"
GRID_COLOR = "#E7E7E7"
LIGHT_GRID_COLOR = "#F1F1F1"
LEGEND_EDGE_COLOR = "#B8B8B8"
SIGN_CORRECT_COLOR = "#4C78A8"
SIGN_INCORRECT_COLOR = "#B55A30"
SIGN_ZERO_COLOR = "#666666"
SIGN_ZERO_ATOL = 1.0e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze one selected AAReact ML model.")
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--hyper-split", default=DEFAULT_HYPER_SPLIT)
    parser.add_argument("--search-method", default=DEFAULT_SEARCH_METHOD)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--descriptor", default=DEFAULT_DESCRIPTOR)
    parser.add_argument("--config", type=Path, default=None, help="Train TOML. If omitted, infer from target/split/model.")
    parser.add_argument("--model", type=Path, default=None, help="Model pickle. Overrides TOML model_save.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Dataset directory. Overrides TOML data paths.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--primary-seed", type=int, default=1)
    parser.add_argument("--primary-test-size", type=float, default=0.2)
    parser.add_argument("--max-shap-samples", type=int, default=0, help="0 uses the full training split.")
    parser.add_argument("--max-display", type=int, default=20)
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=None,
        help="Raw full_data CSV with DATA_ID/TEMP/EE/DDG. If omitted, use the latest DataSet/Data_All/full_data_*.csv.",
    )
    return parser.parse_args()


def apply_style() -> None:
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
        "svg.fonttype": "none",
    })


def infer_config(args: argparse.Namespace) -> Path:
    if args.config is not None:
        return args.config
    return (
        PROJECT_ROOT
        / "config"
        / "train"
        / args.target
        / "hyper_{}".format(args.hyper_split)
        / "search_{}".format(args.search_method)
        / args.split
        / "train_ml_{}_{}.toml".format(args.model_name, args.descriptor)
    )


def load_train_config(config_fp: Path) -> dict:
    if not config_fp.exists():
        raise FileNotFoundError("Train config does not exist: {}".format(config_fp))
    with config_fp.open("rb") as handle:
        return tomllib.load(handle)


def dataset_paths(args: argparse.Namespace, config: dict) -> dict[str, Path]:
    train_cfg = config["Train"]
    if args.data_dir is not None:
        data_dir = args.data_dir
    else:
        data_dir = Path(train_cfg["data_x"]).parent
    paths = {
        "data_x": data_dir / "train_test_data_x.npy",
        "data_y": data_dir / "train_test_data_y.npy",
        "x_label": data_dir / "train_test_x_label.pkl",
        "data_class": data_dir / "train_test_data_class.pkl",
        "data_name": data_dir / "train_test_data_name.pkl",
    }
    if args.data_dir is None:
        paths.update({
            "data_x": Path(train_cfg["data_x"]),
            "data_y": Path(train_cfg["data_y"]),
            "x_label": Path(train_cfg["x_label"]),
            "data_class": Path(train_cfg["data_class"]),
        })
    return paths


def load_pickle(fp: Path):
    with fp.open("rb") as handle:
        return pickle.load(handle)


def load_dataset(paths: dict[str, Path]) -> tuple[np.ndarray, np.ndarray, dict, list, list]:
    for key, fp in paths.items():
        if key == "data_name" and not fp.exists():
            continue
        if not fp.exists():
            raise FileNotFoundError("{} does not exist: {}".format(key, fp))
    data_x = np.load(paths["data_x"])
    data_y = np.load(paths["data_y"]).ravel()
    x_label = load_pickle(paths["x_label"])
    data_class = load_pickle(paths["data_class"])
    if paths["data_name"].exists():
        data_name = load_pickle(paths["data_name"])
    else:
        data_name = ["row_{:05d}".format(idx) for idx in range(len(data_y))]
    if data_x.shape[0] != len(data_y):
        raise ValueError("X/y row mismatch: {} vs {}".format(data_x.shape[0], len(data_y)))
    return data_x, data_y, x_label, data_class, data_name


def label_sort_key(key: str) -> tuple[int, str]:
    match = re.match(r"label(\d+)_(.+)", str(key))
    if match:
        return int(match.group(1)), match.group(2)
    return 999, str(key)


def format_descriptor_label(descriptor: str) -> str:
    label_map = {
        "rdkit": "RDKit",
        "xtb": "xTB",
        "acsf": "ACSF",
        "soap": "SOAP",
    }
    return "+".join(label_map.get(part, part.upper()) for part in str(descriptor).split("_"))


def classify_feature_source(label: str, default_source: str) -> str:
    if str(label).upper() in {"TEMP", "PRESSURE"}:
        return "condition"
    return str(default_source)


def flatten_feature_labels(x_label: dict, n_features: int) -> tuple[list[str], list[str]]:
    labels: list[str] = []
    sources: list[str] = []
    for key in sorted(x_label.keys(), key=label_sort_key):
        source = str(key).split("_", 1)[1] if "_" in str(key) else "other"
        values = list(x_label[key])
        labels.extend(values)
        sources.extend([classify_feature_source(value, source) for value in values])
    if len(labels) != n_features:
        raise ValueError("Feature label count mismatch: labels={}, X columns={}".format(len(labels), n_features))
    return labels, sources


def split_train_test(
    data_x: np.ndarray,
    data_y: np.ndarray,
    data_class: list,
    data_name: list,
    seed: int,
    test_size: float,
):
    return train_test_split(
        data_x,
        data_y,
        data_class,
        data_name,
        test_size=float(test_size),
        random_state=int(seed),
    )


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MSE": mse,
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mse)),
    }


def classify_sign_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape:
        raise ValueError("Sign-classification shape mismatch: {} vs {}".format(true.shape, pred.shape))

    status = np.full(true.shape, "invalid", dtype=object)
    finite = np.isfinite(true) & np.isfinite(pred)
    true_zero = finite & np.isclose(true, 0.0, rtol=0.0, atol=SIGN_ZERO_ATOL)
    evaluable = finite & ~true_zero
    correct = evaluable & (np.sign(true) == np.sign(pred))
    status[evaluable & ~correct] = "incorrect"
    status[correct] = "correct"
    status[true_zero] = "true_zero"
    return status


def calc_sign_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int | float | None]:
    status = classify_sign_predictions(y_true, y_pred)
    n_correct = int(np.count_nonzero(status == "correct"))
    n_incorrect = int(np.count_nonzero(status == "incorrect"))
    n_evaluable = n_correct + n_incorrect
    return {
        "n_total": int(status.size),
        "n_evaluable_nonzero": n_evaluable,
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "n_true_zero_excluded": int(np.count_nonzero(status == "true_zero")),
        "n_invalid": int(np.count_nonzero(status == "invalid")),
        "accuracy_nonzero": float(n_correct / n_evaluable) if n_evaluable else None,
    }


def calc_ddg_from_ee(temp_c: np.ndarray, ee: np.ndarray) -> np.ndarray:
    temp_k = np.asarray(temp_c, dtype=float) + CELSIUS_TO_KELVIN
    ee_arr = np.asarray(ee, dtype=float)
    return DDG_R_KCAL * temp_k * np.log((1.0 + ee_arr) / (1.0 - ee_arr))


def ddg_to_ee(ddg: np.ndarray, temp_c: np.ndarray) -> np.ndarray:
    temp_k = np.asarray(temp_c, dtype=float) + CELSIUS_TO_KELVIN
    return np.tanh(np.asarray(ddg, dtype=float) / (2.0 * DDG_R_KCAL * temp_k))


def infer_source_csv(source_csv: Path | None) -> Path | None:
    if source_csv is not None:
        return source_csv
    candidates = sorted((PROJECT_ROOT / "DataSet" / "Data_All").glob("full_data_*.csv"))
    if not candidates:
        return None
    return candidates[-1]


def load_source_data(source_csv: Path | None) -> tuple[pd.DataFrame | None, dict[str, object]]:
    fp = infer_source_csv(source_csv)
    if fp is None:
        return None, {"source_csv": None, "source_available": False}
    if not fp.exists():
        raise FileNotFoundError("Source CSV does not exist: {}".format(fp))
    df = pd.read_csv(fp)
    required = ["DATA_ID", "TEMP", "EE", "DDG"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError("{} is missing required columns: {}".format(fp, ", ".join(missing)))
    for col in ["TEMP", "EE", "DDG"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    valid = df[required].dropna().copy()
    valid = valid[np.abs(valid["EE"].to_numpy(dtype=float)) < 1.0].copy()
    ddg_calc = calc_ddg_from_ee(valid["TEMP"].to_numpy(dtype=float), valid["EE"].to_numpy(dtype=float))
    diff = valid["DDG"].to_numpy(dtype=float) - ddg_calc
    opposite_mask = (
        (np.abs(valid["EE"].to_numpy(dtype=float)) > 1.0e-12)
        & (np.abs(valid["DDG"].to_numpy(dtype=float)) > 1.0e-12)
        & (np.sign(valid["EE"].to_numpy(dtype=float)) == -np.sign(valid["DDG"].to_numpy(dtype=float)))
    )
    meta = {
        "source_csv": str(fp),
        "source_available": True,
        "source_n": int(len(df)),
        "formula_n_valid": int(len(valid)),
        "ddg_formula": "DDG = R*(TEMP+273.15)*ln((1+EE)/(1-EE))",
        "ddg_r_kcal": DDG_R_KCAL,
        "max_abs_ddg_formula_error": float(np.nanmax(np.abs(diff))) if len(diff) else None,
        "mean_abs_ddg_formula_error": float(np.nanmean(np.abs(diff))) if len(diff) else None,
        "opposite_sign_ee_ddg_count": int(opposite_mask.sum()),
    }
    return df, meta


def save_predictions(
    output_dir: Path,
    name_train: list,
    name_test: list,
    class_train: list,
    class_test: list,
    y_train: np.ndarray,
    y_test: np.ndarray,
    train_pred: np.ndarray,
    test_pred: np.ndarray,
    source_df: pd.DataFrame | None,
) -> Path:
    source_by_id = {}
    if source_df is not None:
        source_for_lookup = source_df.copy()
        source_for_lookup["DATA_ID"] = source_for_lookup["DATA_ID"].astype(str)
        source_by_id = source_for_lookup.drop_duplicates("DATA_ID").set_index("DATA_ID").to_dict(orient="index")
    rows = []
    for split, names, classes, y_true, y_pred in (
        ("train", name_train, class_train, y_train, train_pred),
        ("test", name_test, class_test, y_test, test_pred),
    ):
        sign_status = classify_sign_predictions(y_true, y_pred)
        for name, cls, yt, yp, status in zip(names, classes, y_true, y_pred, sign_status):
            row = {
                "split": split,
                "DATA_ID": name,
                "class": cls,
                "y_true": float(yt),
                "y_pred": float(yp),
                "residual": float(yp - yt),
                "abs_error": float(abs(yp - yt)),
                "target_sign_status": status,
                "target_sign_correct": True if status == "correct" else False if status == "incorrect" else None,
            }
            source_row = source_by_id.get(str(name))
            if source_row is not None and pd.notna(source_row.get("TEMP")):
                temp_c = float(source_row["TEMP"])
                row["temperature_c"] = temp_c
                if pd.notna(source_row.get("EE")):
                    row["true_ee"] = float(source_row["EE"])
                if pd.notna(source_row.get("DDG")):
                    row["source_ddg"] = float(source_row["DDG"])
                row["pred_ee"] = float(ddg_to_ee(np.asarray([yp], dtype=float), np.asarray([temp_c], dtype=float))[0])
                if "true_ee" in row:
                    row["ee_sign_correct"] = bool(np.sign(row["true_ee"]) == np.sign(row["pred_ee"]))
            rows.append(row)
    out_fp = output_dir / "selected_model_train_test_predictions.csv"
    pd.DataFrame(rows).to_csv(out_fp, index=False)
    return out_fp


def fit_line(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float] | None:
    if len(y_true) < 2:
        return None
    return tuple(float(x) for x in np.polyfit(y_true, y_pred, deg=1))


def format_fit_equation(split: str, coeff: tuple[float, float]) -> str:
    slope, intercept = coeff
    sign = "+" if intercept >= 0 else "-"
    return "{} fit: y = {:.3f}x {} {:.3f}".format(split, slope, sign, abs(intercept))


def format_metrics_label(split: str, metrics: dict[str, float], n: int) -> str:
    return "{} (n={}, RMSE={:.3f}, R2={:.3f})".format(split, int(n), metrics["RMSE"], metrics["R2"])


def add_ee_sign_quadrants(ax, lims: tuple[float, float], show_labels: bool = True) -> None:
    low, high = lims
    if not (low < 0.0 < high):
        return

    correct_color = "#EEF7F1"
    flip_color = "#FFF1EF"
    background_alpha = 0.58
    ax.add_patch(plt.Rectangle((low, low), -low, -low, facecolor=correct_color, edgecolor="none", alpha=background_alpha, zorder=0))
    ax.add_patch(plt.Rectangle((0.0, 0.0), high, high, facecolor=correct_color, edgecolor="none", alpha=background_alpha, zorder=0))
    ax.add_patch(plt.Rectangle((low, 0.0), -low, high, facecolor=flip_color, edgecolor="none", alpha=background_alpha, zorder=0))
    ax.add_patch(plt.Rectangle((0.0, low), high, -low, facecolor=flip_color, edgecolor="none", alpha=background_alpha, zorder=0))
    ax.axvline(0.0, color="#8A8A8A", linestyle=(0, (2.0, 2.0)), linewidth=0.72, alpha=0.78, zorder=1.2)
    ax.axhline(0.0, color="#8A8A8A", linestyle=(0, (2.0, 2.0)), linewidth=0.72, alpha=0.78, zorder=1.2)

    if not show_labels:
        return

    span = high - low
    labels = [
        (-0.08 * (-low), 0.48 * high, "- / +\nflip", "center", "center"),
        (0.82 * high, 0.58 * high, "+ / +\ncorrect", "center", "center"),
        (0.38 * low, 0.86 * low, "- / -\ncorrect", "center", "center"),
        (0.36 * high, 0.38 * low, "+ / -\nflip", "center", "center"),
    ]
    for x_pos, y_pos, label, ha, va in labels:
        ax.text(
            x_pos,
            y_pos,
            label,
            ha=ha,
            va=va,
            fontsize=6.0,
            color="#555555",
            linespacing=1.0,
            alpha=0.66,
            zorder=1.8,
        )


def plot_regression_diagnostics(
    output_dir: Path,
    y_train: np.ndarray,
    y_test: np.ndarray,
    train_pred: np.ndarray,
    test_pred: np.ndarray,
    train_metrics: dict[str, float],
    test_metrics: dict[str, float],
    title: str,
) -> list[str]:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.25),
        dpi=300,
        gridspec_kw={"width_ratios": [1.18, 0.82]},
    )
    ax, ax_res = axes
    fig.patch.set_facecolor("white")
    for one_ax in axes:
        one_ax.set_facecolor("white")
        for spine in one_ax.spines.values():
            spine.set_visible(True)
            spine.set_color(AXIS_COLOR)
            spine.set_linewidth(0.8)
        one_ax.tick_params(axis="both", direction="out", top=False, right=False, width=0.8)

    all_values = np.concatenate([y_train, y_test, train_pred, test_pred])
    low = float(np.nanmin(all_values))
    high = float(np.nanmax(all_values))
    pad = 0.08 * max(high - low, 1.0e-6)
    lims = (low - pad, high + pad)
    line_x = np.linspace(lims[0], lims[1], 120)

    add_ee_sign_quadrants(ax, lims)
    ax.plot(line_x, line_x, color=LEGEND_EDGE_COLOR, linestyle="--", linewidth=0.9, label="Ideal")
    fit_annotations: list[tuple[str, str]] = []
    for split, y_true, y_pred, color, marker, metrics in (
        ("Train", y_train, train_pred, MODEL_COLORS["train"], "o", train_metrics),
        ("Test", y_test, test_pred, MODEL_COLORS["test"], "s", test_metrics),
    ):
        ax.scatter(
            y_true,
            y_pred,
            s=22 if split == "Train" else 28,
            marker=marker,
            facecolor=color,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.78 if split == "Train" else 0.90,
            label=split,
            zorder=3,
        )
        coeff = fit_line(y_true, y_pred)
        if coeff is not None:
            slope, intercept = coeff
            ax.plot(line_x, slope * line_x + intercept, color=color, linewidth=1.0, alpha=0.78)
            fit_annotations.append((format_fit_equation(split, coeff), color))

    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Experimental $\Delta\Delta G$ (kcal mol$^{-1}$)")
    ax.set_ylabel(r"Predicted $\Delta\Delta G$ (kcal mol$^{-1}$)")
    ax.set_title("Prediction parity", pad=6)
    ax.grid(axis="both", color=GRID_COLOR, linewidth=0.42)
    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=True,
        framealpha=0.94,
        facecolor="white",
        edgecolor=LEGEND_EDGE_COLOR,
        borderpad=0.25,
        handletextpad=0.40,
        labelspacing=0.35,
    )
    legend.get_frame().set_linewidth(0.55)
    for idx, (text, color) in enumerate(fit_annotations):
        ax.text(
            0.97,
            0.055 + 0.070 * (len(fit_annotations) - idx - 1),
            text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.6,
            color=color,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.74, "pad": 1.4},
            zorder=6,
        )

    train_res = train_pred - y_train
    test_res = test_pred - y_test
    bins = np.linspace(
        float(min(train_res.min(), test_res.min())),
        float(max(train_res.max(), test_res.max())),
        18,
    )
    ax_res.axvline(0.0, color=LEGEND_EDGE_COLOR, linestyle="--", linewidth=0.85)
    ax_res.hist(train_res, bins=bins, density=True, histtype="stepfilled", color=MODEL_COLORS["train"], alpha=0.28)
    ax_res.hist(test_res, bins=bins, density=True, histtype="stepfilled", color=MODEL_COLORS["test"], alpha=0.34)
    ax_res.hist(train_res, bins=bins, density=True, histtype="step", color=MODEL_COLORS["train"], linewidth=1.1, label="Train")
    ax_res.hist(test_res, bins=bins, density=True, histtype="step", color=MODEL_COLORS["test"], linewidth=1.1, label="Test")
    ax_res.set_xlabel("Residual (predicted - experimental)")
    ax_res.set_ylabel("Density")
    ax_res.set_title("Residual distribution", pad=6)
    ax_res.grid(axis="y", color=GRID_COLOR, linewidth=0.42)
    ax_res.legend(frameon=False, loc="upper right", handlelength=1.2)

    fig.suptitle(title, y=0.995, fontsize=9)
    fig.text(
        0.50,
        0.028,
        "{}; {}".format(
            format_metrics_label("Train", train_metrics, len(y_train)),
            format_metrics_label("Test", test_metrics, len(y_test)),
        ),
        ha="center",
        va="center",
        fontsize=7.2,
        color=AXIS_COLOR,
    )
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.94), w_pad=1.1)
    outputs = []
    for suffix in ("png", "svg"):
        fp = output_dir / "selected_model_train_test_regression.{}".format(suffix)
        fig.savefig(fp, bbox_inches="tight")
        outputs.append(str(fp))
    plt.close(fig)
    return outputs


def plot_sign_accuracy_by_split(
    output_dir: Path,
    y_train: np.ndarray,
    y_test: np.ndarray,
    train_pred: np.ndarray,
    test_pred: np.ndarray,
    title: str,
) -> tuple[list[str], dict[str, dict[str, int | float | None]]]:
    datasets = (
        ("Train", np.asarray(y_train, dtype=float), np.asarray(train_pred, dtype=float)),
        ("Test", np.asarray(y_test, dtype=float), np.asarray(test_pred, dtype=float)),
    )
    sign_metrics = {
        split.lower(): calc_sign_metrics(y_true, y_pred)
        for split, y_true, y_pred in datasets
    }

    all_true = np.concatenate([y_train, y_test])
    y_low = float(np.nanmin(all_true))
    y_high = float(np.nanmax(all_true))
    y_span = max(y_high - y_low, 1.0e-6)

    fig, ax = plt.subplots(figsize=(5.4, 3.6), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(AXIS_COLOR)
        spine.set_linewidth(0.8)
    ax.tick_params(axis="both", direction="out", top=False, right=False, width=0.8)

    boxplot = ax.boxplot(
        [y_train, y_test],
        positions=[0.0, 1.0],
        widths=0.30,
        patch_artist=True,
        showfliers=False,
        boxprops={"edgecolor": "#666666", "linewidth": 0.8},
        medianprops={"color": AXIS_COLOR, "linewidth": 1.0},
        whiskerprops={"color": "#888888", "linewidth": 0.75},
        capprops={"color": "#888888", "linewidth": 0.75},
    )
    for box in boxplot["boxes"]:
        box.set_facecolor("#F7F7F7")
        box.set_alpha(0.88)
        box.set_zorder(1.5)

    rng = np.random.default_rng(20260804)
    point_specs = {
        "correct": {
            "s": 15,
            "marker": "o",
            "facecolor": SIGN_CORRECT_COLOR,
            "edgecolor": "white",
            "linewidth": 0.22,
            "alpha": 0.48,
            "label": "Sign correct",
            "zorder": 3,
        },
        "incorrect": {
            "s": 22,
            "marker": "X",
            "color": SIGN_INCORRECT_COLOR,
            "linewidth": 0.50,
            "alpha": 0.94,
            "label": "Sign incorrect",
            "zorder": 4,
        },
        "true_zero": {
            "s": 14,
            "marker": "D",
            "facecolor": "white",
            "edgecolor": SIGN_ZERO_COLOR,
            "linewidth": 0.50,
            "alpha": 0.82,
            "label": r"Recorded $\Delta\Delta G=0$",
            "zorder": 3.5,
        },
    }
    for split_idx, (split, y_true, y_pred) in enumerate(datasets):
        x_jitter = split_idx + rng.uniform(-0.16, 0.16, size=len(y_true))
        status = classify_sign_predictions(y_true, y_pred)
        for sign_status, style in point_specs.items():
            mask = status == sign_status
            scatter_style = dict(style)
            if split_idx > 0:
                scatter_style["label"] = None
            ax.scatter(x_jitter[mask], y_true[mask], **scatter_style)

        metrics = sign_metrics[split.lower()]
        accuracy = metrics["accuracy_nonzero"]
        ax.text(
            split_idx,
            y_high + 0.11 * y_span,
            "{}/{} correct ({:.1%})".format(
                metrics["n_correct"],
                metrics["n_evaluable_nonzero"],
                float(accuracy),
            ) if accuracy is not None else "Sign accuracy not defined",
            ha="center",
            va="center",
            fontsize=7.0,
            color=AXIS_COLOR,
        )

    ax.axhline(0.0, color="#8A8A8A", linestyle=(0, (2.0, 2.0)), linewidth=0.75, alpha=0.82, zorder=1)
    ax.set_xlim(-0.44, 1.44)
    ax.set_ylim(y_low - 0.07 * y_span, y_high + 0.19 * y_span)
    ax.set_xticks([0.0, 1.0], ["Train (n={})".format(len(y_train)), "Test (n={})".format(len(y_test))])
    ax.set_ylabel(r"Reference $\Delta\Delta G$ (kcal mol$^{-1}$)")
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.42)
    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=3,
        frameon=True,
        framealpha=0.94,
        facecolor="white",
        edgecolor=LEGEND_EDGE_COLOR,
        borderpad=0.35,
        handletextpad=0.45,
        labelspacing=0.40,
        columnspacing=1.25,
        markerscale=0.90,
    )
    legend.get_frame().set_linewidth(0.55)
    fig.suptitle("Sign correctness by data split", y=0.975, fontsize=9)
    fig.text(0.5, 0.915, title, ha="center", va="center", fontsize=7.0, color="#555555")
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.25, top=0.79)

    outputs = []
    for suffix in ("png", "svg"):
        fp = output_dir / "selected_model_test_sign_accuracy.{}".format(suffix)
        fig.savefig(fp, bbox_inches="tight")
        outputs.append(str(fp))
    plt.close(fig)
    return outputs, sign_metrics


def sample_for_shap(X_train: np.ndarray, max_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if int(max_samples) <= 0 or X_train.shape[0] <= int(max_samples):
        idx = np.arange(X_train.shape[0])
        return X_train, idx
    rng = np.random.default_rng(int(seed))
    idx = np.sort(rng.choice(X_train.shape[0], size=int(max_samples), replace=False))
    return X_train[idx], idx


def compute_shap_values(model, X_shap: np.ndarray) -> np.ndarray:
    import shap

    explainer = shap.TreeExplainer(model)
    try:
        values = explainer.shap_values(X_shap, check_additivity=False)
    except TypeError:
        values = explainer.shap_values(X_shap)
    if isinstance(values, list):
        values = values[0]
    if hasattr(values, "values"):
        values = values.values
    return np.asarray(values, dtype=float)


def save_shap_importance(
    output_dir: Path,
    shap_values: np.ndarray,
    feature_labels: list[str],
    feature_sources: list[str],
) -> pd.DataFrame:
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({
        "feature": feature_labels,
        "source": feature_sources,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    df.to_csv(output_dir / "selected_model_shap_feature_importance.csv", index=False)
    return df


def shorten_label(label: str, max_len: int = 34) -> str:
    text = str(label)
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "..."


def format_source_label(source: str) -> str:
    label_map = {
        "rdkit": "RDKit",
        "xtb": "xTB",
        "acsf": "ACSF",
        "soap": "SOAP",
        "condition": "Condition",
    }
    return label_map.get(str(source), str(source))


def classify_feature_role(label: str) -> str:
    text = str(label).upper()
    if text == "PRESSURE":
        return "pressure"
    if text == "TEMP":
        return "temperature"
    if text.startswith("CAT_"):
        return "catalyst"
    if text.startswith("REA_"):
        return "reactant"
    if text.startswith("SOL_"):
        return "solvent"
    return "other"


def format_role_label(role: str) -> str:
    label_map = {
        "catalyst": "Catalyst features",
        "reactant": "Reactant features",
        "solvent": "Solvent features",
        "pressure": "Pressure",
        "temperature": "Temperature",
        "other": "Other",
    }
    return label_map.get(str(role), str(role))


def summarize_shap_contributions(output_dir: Path, importance: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    total = float(importance["mean_abs_shap"].sum())
    if total <= 0.0:
        raise ValueError("SHAP contribution total is non-positive.")

    source_summary = (
        importance.groupby("source", as_index=False)["mean_abs_shap"]
        .sum()
        .sort_values("mean_abs_shap", ascending=False)
    )
    source_summary["label"] = source_summary["source"].map(format_source_label)
    source_summary["percent"] = 100.0 * source_summary["mean_abs_shap"] / total

    role_df = importance.copy()
    role_df["role"] = role_df["feature"].map(classify_feature_role)
    role_summary = (
        role_df.groupby("role", as_index=False)["mean_abs_shap"]
        .sum()
        .sort_values("mean_abs_shap", ascending=False)
    )
    role_summary["label"] = role_summary["role"].map(format_role_label)
    role_summary["percent"] = 100.0 * role_summary["mean_abs_shap"] / total

    rows = []
    for summary_name, frame, key_col in (
        ("descriptor_source", source_summary, "source"),
        ("molecular_role", role_summary, "role"),
    ):
        for _, row in frame.iterrows():
            rows.append({
                "summary": summary_name,
                "group": row[key_col],
                "label": row["label"],
                "mean_abs_shap_sum": float(row["mean_abs_shap"]),
                "percent": float(row["percent"]),
            })
    out_fp = output_dir / "selected_model_shap_contribution_summary.csv"
    pd.DataFrame(rows).to_csv(out_fp, index=False)
    return source_summary, role_summary, out_fp


def plot_shap_contribution_summary(
    output_dir: Path,
    source_summary: pd.DataFrame,
    role_summary: pd.DataFrame,
) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.15), dpi=300)
    fig.patch.set_facecolor("white")
    panels = [
        (axes[0], source_summary, "source", SOURCE_COLORS, "Descriptor source"),
        (axes[1], role_summary, "role", ROLE_COLORS, "Molecular role / condition"),
    ]
    for panel_label, (ax, frame, key_col, palette, title) in zip(["A", "B"], panels):
        plot_df = frame.sort_values("percent", ascending=True).reset_index(drop=True)
        y_pos = np.arange(len(plot_df))
        colors = [palette.get(str(item), palette.get("other", "#777777")) for item in plot_df[key_col]]
        ax.barh(
            y_pos,
            plot_df["percent"],
            color=colors,
            alpha=0.88,
            edgecolor=AXIS_COLOR,
            linewidth=0.25,
            height=0.68,
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_df["label"])
        ax.set_xlabel("Contribution to total mean |SHAP| (%)")
        ax.set_title(title, pad=6)
        ax.grid(axis="x", color=GRID_COLOR, linewidth=0.42)
        ax.set_xlim(0.0, max(5.0, float(plot_df["percent"].max()) * 1.18))
        for idx, value in enumerate(plot_df["percent"]):
            ax.text(
                float(value) + 0.8,
                idx,
                "{:.1f}%".format(float(value)),
                va="center",
                ha="left",
                fontsize=7.0,
                color=AXIS_COLOR,
            )
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(AXIS_COLOR)
            spine.set_linewidth(0.8)
        ax.tick_params(axis="both", direction="out", top=False, right=False, width=0.8)
        ax.text(
            -0.12,
            1.04,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=AXIS_COLOR,
        )
    fig.tight_layout(w_pad=2.0, pad=0.45)
    outputs = []
    for suffix in ("png", "svg"):
        fp = output_dir / "selected_model_shap_contribution_summary.{}".format(suffix)
        fig.savefig(fp, bbox_inches="tight")
        outputs.append(str(fp))
    plt.close(fig)
    return outputs


def plot_ee_ddg_mapping(output_dir: Path, source_df: pd.DataFrame | None) -> list[str]:
    if source_df is None:
        return []
    required = ["TEMP", "EE", "DDG"]
    if any(col not in source_df.columns for col in required):
        return []
    df = source_df[required].copy()
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()
    df = df[np.abs(df["EE"].to_numpy(dtype=float)) < 1.0].copy()
    if df.empty:
        return []

    fig, ax = plt.subplots(figsize=(4.85, 3.15), dpi=300)
    fig.patch.set_facecolor("white")
    temp_values = sorted(float(item) for item in df["TEMP"].dropna().unique())
    ee_grid = np.linspace(-0.98, 0.98, 600)
    line_colors = {
        40.0: "#4C78A8",
        50.0: "#2F8F83",
        60.0: "#B55A30",
    }
    for temp in temp_values:
        color = line_colors.get(temp, "#777777")
        temp_df = df[np.isclose(df["TEMP"].to_numpy(dtype=float), temp)]
        ax.plot(
            ee_grid,
            calc_ddg_from_ee(np.full_like(ee_grid, temp), ee_grid),
            color=color,
            linewidth=1.05,
            alpha=0.92,
            label="{:.0f} deg C".format(temp),
        )
        ax.scatter(
            temp_df["EE"],
            temp_df["DDG"],
            color=color,
            s=13,
            alpha=0.68,
            edgecolor="white",
            linewidth=0.22,
            zorder=3,
        )
    ax.axhline(0.0, color=LEGEND_EDGE_COLOR, linestyle=(0, (2.0, 2.0)), linewidth=0.8)
    ax.axvline(0.0, color=LEGEND_EDGE_COLOR, linestyle=(0, (2.0, 2.0)), linewidth=0.8)
    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(
        float(min(-3.05, df["DDG"].min() - 0.25)),
        float(max(3.05, df["DDG"].max() + 0.25)),
    )
    ax.set_xlabel(r"Experimental ee = (R - S) / (R + S)")
    ax.set_ylabel(r"$\Delta\Delta G = \Delta G_S^\ddagger - \Delta G_R^\ddagger$ (kcal mol$^{-1}$)")
    ax.set_title("ee-ddG sign convention", pad=5)
    ax.grid(axis="both", color=GRID_COLOR, linewidth=0.42)
    ax.text(
        0.96,
        0.89,
        "R-major\npositive ee\npositive ddG",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.7,
        color="#555555",
        linespacing=1.12,
    )
    ax.text(
        0.04,
        0.20,
        "S-major\nnegative ee\nnegative ddG",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.7,
        color="#555555",
        linespacing=1.12,
    )
    ax.text(
        0.04,
        0.08,
        r"$\Delta\Delta G=RT\ln\frac{1+ee}{1-ee}$",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.8,
        color=AXIS_COLOR,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.6},
    )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        legend = ax.legend(
            handles,
            labels,
            title="Formula curve",
            frameon=True,
            framealpha=0.94,
            facecolor="white",
            edgecolor=LEGEND_EDGE_COLOR,
            loc="upper right",
            borderpad=0.28,
            handlelength=1.5,
            labelspacing=0.28,
        )
        legend.get_frame().set_linewidth(0.55)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(AXIS_COLOR)
        spine.set_linewidth(0.8)
    ax.tick_params(axis="both", direction="out", top=False, right=False, width=0.8)
    fig.tight_layout(pad=0.45)
    outputs = []
    for suffix in ("png", "svg"):
        fp = output_dir / "selected_model_ee_ddg_mapping.{}".format(suffix)
        fig.savefig(fp, bbox_inches="tight")
        outputs.append(str(fp))
    plt.close(fig)
    return outputs


def plot_shap_bar(output_dir: Path, importance: pd.DataFrame, max_display: int) -> list[str]:
    top = importance.head(int(max_display)).iloc[::-1].copy()
    colors = [SOURCE_COLORS.get(str(src), SOURCE_COLORS["other"]) for src in top["source"]]
    fig_h = max(3.4, 0.22 * len(top) + 1.15)
    fig, ax = plt.subplots(figsize=(5.45, fig_h), dpi=300)
    ax.barh(np.arange(len(top)), top["mean_abs_shap"], color=colors, alpha=0.86, edgecolor=AXIS_COLOR, linewidth=0.25)
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels([shorten_label(item) for item in top["feature"]])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_ylabel("Feature")
    ax.set_title("Global SHAP importance", pad=6)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.42)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(AXIS_COLOR)
        spine.set_linewidth(0.8)
    ax.tick_params(axis="both", direction="out", top=False, right=False, width=0.8)

    present_sources = []
    for src in top["source"]:
        if src not in present_sources:
            present_sources.append(src)
    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="none", markersize=5.0,
                   markerfacecolor=SOURCE_COLORS.get(src, SOURCE_COLORS["other"]),
                   markeredgecolor=AXIS_COLOR, markeredgewidth=0.25, label=format_source_label(str(src)))
        for src in present_sources
    ]
    ax.legend(handles=handles, title="Source", frameon=False, loc="lower right")
    fig.tight_layout(pad=0.35)
    outputs = []
    for suffix in ("png", "svg"):
        fp = output_dir / "selected_model_shap_importance.{}".format(suffix)
        fig.savefig(fp, bbox_inches="tight")
        outputs.append(str(fp))
    plt.close(fig)
    return outputs


def plot_shap_beeswarm(
    output_dir: Path,
    shap_values: np.ndarray,
    X_shap: np.ndarray,
    importance: pd.DataFrame,
    feature_labels: list[str],
    max_display: int,
) -> list[str]:
    top_features = importance.head(int(max_display))["feature"].tolist()
    label_to_idx = {label: idx for idx, label in enumerate(feature_labels)}
    features = [feature for feature in top_features if feature in label_to_idx]
    if not features:
        return []

    fig_h = max(3.4, 0.23 * len(features) + 1.35)
    fig, ax = plt.subplots(figsize=(5.8, fig_h), dpi=300)
    rng = np.random.default_rng(0)
    cmap = plt.get_cmap("coolwarm")
    y_positions = np.arange(len(features), dtype=float)

    for y_pos, feature in zip(y_positions, features):
        idx = label_to_idx[feature]
        values = X_shap[:, idx].astype(float)
        shap_col = shap_values[:, idx].astype(float)
        finite_values = values[np.isfinite(values)]
        if len(finite_values) == 0:
            normed = np.full_like(values, 0.5, dtype=float)
        else:
            low, high = np.percentile(finite_values, [5, 95])
            if abs(high - low) < 1.0e-12:
                normed = np.full_like(values, 0.5, dtype=float)
            else:
                normed = np.clip((values - low) / (high - low), 0.0, 1.0)
        jitter = rng.uniform(-0.17, 0.17, size=len(shap_col))
        ax.scatter(
            shap_col,
            np.full_like(shap_col, y_pos, dtype=float) + jitter,
            c=normed,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            s=12,
            alpha=0.78,
            edgecolor="none",
            rasterized=True,
            zorder=3,
        )

    ax.axvline(0.0, color=LEGEND_EDGE_COLOR, linewidth=0.85, zorder=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([shorten_label(feature) for feature in features])
    ax.invert_yaxis()
    ax.set_xlabel("SHAP value")
    ax.set_ylabel("Feature")
    ax.set_title("SHAP beeswarm", pad=6)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.42)
    ax.grid(axis="y", color=LIGHT_GRID_COLOR, linewidth=0.32)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(AXIS_COLOR)
        spine.set_linewidth(0.8)
    ax.tick_params(axis="both", direction="out", top=False, right=False, width=0.8)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.78, pad=0.025)
    cbar.set_ticks([0.0, 1.0])
    cbar.set_ticklabels(["Low", "High"])
    cbar.set_label("Feature value")
    cbar.outline.set_linewidth(0.55)
    fig.tight_layout(pad=0.35)
    outputs = []
    for suffix in ("png", "svg"):
        fp = output_dir / "selected_model_shap_beeswarm.{}".format(suffix)
        fig.savefig(fp, bbox_inches="tight")
        outputs.append(str(fp))
    plt.close(fig)
    return outputs


def plot_shap_dependence(
    output_dir: Path,
    X_shap: np.ndarray,
    shap_values: np.ndarray,
    importance: pd.DataFrame,
    feature_labels: list[str],
    max_panels: int = 4,
) -> list[str]:
    top_features = importance.head(int(max_panels))["feature"].tolist()
    if not top_features:
        return []
    label_to_idx = {label: idx for idx, label in enumerate(feature_labels)}
    n_panels = len(top_features)
    n_cols = 2 if n_panels > 1 else 1
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.0, 2.35 * n_rows), dpi=300)
    axes_array = np.asarray(axes).reshape(-1)
    for ax, feature in zip(axes_array, top_features):
        idx = label_to_idx[feature]
        ax.scatter(
            X_shap[:, idx],
            shap_values[:, idx],
            s=16,
            color=MODEL_COLORS["test"],
            alpha=0.72,
            edgecolor="white",
            linewidth=0.20,
        )
        ax.axhline(0.0, color=LEGEND_EDGE_COLOR, linestyle="--", linewidth=0.75)
        ax.set_xlabel(shorten_label(feature, 28))
        ax.set_ylabel("SHAP value")
        ax.grid(axis="both", color=GRID_COLOR, linewidth=0.36)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(AXIS_COLOR)
            spine.set_linewidth(0.8)
        ax.tick_params(axis="both", direction="out", top=False, right=False, width=0.8)
    for ax in axes_array[n_panels:]:
        ax.axis("off")
    fig.suptitle("Top-feature SHAP dependence", y=0.995, fontsize=9)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95), pad=0.35)
    outputs = []
    for suffix in ("png", "svg"):
        fp = output_dir / "selected_model_shap_dependence_top4.{}".format(suffix)
        fig.savefig(fp, bbox_inches="tight")
        outputs.append(str(fp))
    plt.close(fig)
    return outputs


def write_interpretation_notes(
    output_dir: Path,
    model_label: str,
    seed: int,
    test_size: float,
    train_metrics: dict[str, float],
    test_metrics: dict[str, float],
    sign_metrics: dict[str, dict[str, int | float | None]],
    source_meta: dict[str, object],
    source_summary: pd.DataFrame,
    role_summary: pd.DataFrame,
    importance: pd.DataFrame,
) -> Path:
    def summary_lines(frame: pd.DataFrame, name_col: str) -> list[str]:
        return [
            "- {}: {:.1f}%".format(row["label"], float(row["percent"]))
            for _, row in frame.sort_values("percent", ascending=False).iterrows()
        ]

    top = importance.head(10).copy()
    lines = [
        "# Selected-model interpretation",
        "",
        "## Model",
        "",
        "- Model: `{}`".format(model_label),
        "- Split: `seed={}`, `test_size={:.2f}`".format(int(seed), float(test_size)),
        "- Train RMSE/R2: `{:.4f}` / `{:.4f}`".format(train_metrics["RMSE"], train_metrics["R2"]),
        "- Test RMSE/R2: `{:.4f}` / `{:.4f}`".format(test_metrics["RMSE"], test_metrics["R2"]),
        "",
        "## Sign prediction by split",
        "",
        "- Train: `{}/{}` correct among nonzero targets (`{:.1%}`); true-zero rows excluded: `{}`".format(
            sign_metrics["train"]["n_correct"],
            sign_metrics["train"]["n_evaluable_nonzero"],
            float(sign_metrics["train"]["accuracy_nonzero"] or 0.0),
            sign_metrics["train"]["n_true_zero_excluded"],
        ),
        "- Test: `{}/{}` correct among nonzero targets (`{:.1%}`); true-zero rows excluded: `{}`".format(
            sign_metrics["test"]["n_correct"],
            sign_metrics["test"]["n_evaluable_nonzero"],
            float(sign_metrics["test"]["accuracy_nonzero"] or 0.0),
            sign_metrics["test"]["n_true_zero_excluded"],
        ),
        "",
        "## ee-ddG convention",
        "",
        "The original ee is defined as `(R - S) / (R + S)`. The dataset uses:",
        "",
        "```text",
        "DDG = DeltaG_S^ddagger - DeltaG_R^ddagger",
        "DDG = R * (TEMP + 273.15) * ln((1 + ee) / (1 - ee))",
        "ee = tanh(DDG / (2RT))",
        "```",
        "",
        "Therefore, positive ee corresponds to positive DDG, and negative ee corresponds to negative DDG.",
    ]
    if source_meta.get("source_available"):
        lines.extend([
            "",
            "Formula check from `{}`:".format(source_meta.get("source_csv")),
            "",
            "- Valid rows: `{}`".format(source_meta.get("formula_n_valid")),
            "- Max absolute formula error: `{:.3e}`".format(float(source_meta.get("max_abs_ddg_formula_error") or 0.0)),
            "- Mean absolute formula error: `{:.3e}`".format(float(source_meta.get("mean_abs_ddg_formula_error") or 0.0)),
            "- Opposite-sign non-zero ee/DDG rows: `{}`".format(source_meta.get("opposite_sign_ee_ddg_count")),
        ])

    lines.extend([
        "",
        "## SHAP interpretation rule",
        "",
        "- Positive SHAP value raises DDG and pushes the prediction toward R-major / positive ee.",
        "- Negative SHAP value lowers DDG and pushes the prediction toward S-major / negative ee.",
        "",
        "## Contribution by descriptor source",
        "",
        *summary_lines(source_summary, "source"),
        "",
        "## Contribution by molecular role / condition",
        "",
        *summary_lines(role_summary, "role"),
        "",
        "## Top SHAP features",
        "",
    ])
    for _, row in top.iterrows():
        lines.append("- {}. `{}` ({}, mean |SHAP| = {:.4f})".format(
            int(row["rank"]),
            row["feature"],
            format_source_label(row["source"]),
            float(row["mean_abs_shap"]),
        ))
    out_fp = output_dir / "selected_model_interpretation.md"
    out_fp.write_text("\n".join(lines) + "\n")
    return out_fp


def main() -> None:
    args = parse_args()
    apply_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config_fp = infer_config(args)
    config = load_train_config(config_fp)
    train_cfg = config["Train"]
    seed = int(train_cfg.get("seed", args.primary_seed))
    test_size = float(train_cfg.get("test_size", args.primary_test_size))
    model_fp = args.model if args.model is not None else Path(train_cfg["model_save"])
    if not model_fp.exists():
        raise FileNotFoundError("Model file does not exist: {}".format(model_fp))

    paths = dataset_paths(args, config)
    data_x, data_y, x_label, data_class, data_name = load_dataset(paths)
    feature_labels, feature_sources = flatten_feature_labels(x_label, data_x.shape[1])
    model = load(model_fp)
    source_df, source_meta = load_source_data(args.source_csv)

    X_train, X_test, y_train, y_test, class_train, class_test, name_train, name_test = split_train_test(
        data_x,
        data_y,
        data_class,
        data_name,
        seed,
        test_size,
    )
    train_pred = np.asarray(model.predict(X_train), dtype=float)
    test_pred = np.asarray(model.predict(X_test), dtype=float)
    train_metrics = calc_metrics(y_train, train_pred)
    test_metrics = calc_metrics(y_test, test_pred)
    predictions_fp = save_predictions(
        args.output_dir,
        name_train,
        name_test,
        class_train,
        class_test,
        y_train,
        y_test,
        train_pred,
        test_pred,
        source_df,
    )

    model_label = "{}@{}".format(args.model_name.upper(), format_descriptor_label(args.descriptor))
    thermo_outputs = plot_ee_ddg_mapping(args.output_dir, source_df)
    regression_outputs = plot_regression_diagnostics(
        args.output_dir,
        y_train,
        y_test,
        train_pred,
        test_pred,
        train_metrics,
        test_metrics,
        "{}; seed={} / test={:.2f}".format(model_label, seed, test_size),
    )
    sign_outputs, sign_metrics = plot_sign_accuracy_by_split(
        args.output_dir,
        y_train,
        y_test,
        train_pred,
        test_pred,
        "{}; seed={} / test={:.2f}".format(model_label, seed, test_size),
    )

    X_shap, shap_idx = sample_for_shap(X_train, args.max_shap_samples, seed)
    shap_values = compute_shap_values(model, X_shap)
    importance = save_shap_importance(args.output_dir, shap_values, feature_labels, feature_sources)
    shap_outputs = []
    shap_outputs.extend(plot_shap_bar(args.output_dir, importance, args.max_display))
    shap_outputs.extend(plot_shap_beeswarm(args.output_dir, shap_values, X_shap, importance, feature_labels, args.max_display))
    shap_outputs.extend(plot_shap_dependence(args.output_dir, X_shap, shap_values, importance, feature_labels))
    source_summary, role_summary, contribution_fp = summarize_shap_contributions(args.output_dir, importance)
    shap_outputs.extend(plot_shap_contribution_summary(args.output_dir, source_summary, role_summary))
    notes_fp = write_interpretation_notes(
        args.output_dir,
        model_label,
        seed,
        test_size,
        train_metrics,
        test_metrics,
        sign_metrics,
        source_meta,
        source_summary,
        role_summary,
        importance,
    )

    metadata = {
        "target": args.target,
        "hyper_split": args.hyper_split,
        "search_method": args.search_method,
        "split": args.split,
        "model_name": args.model_name,
        "descriptor": args.descriptor,
        "config": str(config_fp),
        "model": str(model_fp),
        "data_x": str(paths["data_x"]),
        "seed": seed,
        "test_size": test_size,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_features": int(data_x.shape[1]),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "sign_metrics": sign_metrics,
        "test_sign_metrics": sign_metrics["test"],
        "shap_samples": int(X_shap.shape[0]),
        "shap_sample_indices": [] if len(shap_idx) == len(y_train) else [int(item) for item in shap_idx.tolist()],
        "source_data": source_meta,
        "outputs": [
            str(predictions_fp),
            *thermo_outputs,
            *regression_outputs,
            *sign_outputs,
            str(args.output_dir / "selected_model_shap_feature_importance.csv"),
            str(contribution_fp),
            *shap_outputs,
            str(notes_fp),
            str(args.output_dir / "selected_model_analysis_manifest.json"),
        ],
    }
    manifest_fp = args.output_dir / "selected_model_analysis_manifest.json"
    manifest_fp.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
