"""
14_pdp.py — Partial Dependence Plots (one-way and two-way).
Methodology §3.12.3:
  - One-way PDPs: 5 features, 50 quantile-spaced grid points.
  - Two-way PDPs: 2 feature pairs, 30×30 quantile-spaced grid.
  - Computed on TRAINING set (adequate marginal distribution coverage).
  Uses the pre-specified features when retained; otherwise falls back to the
  closest retained physical composite/proxy.
Outputs: figures/pdp_oneway.png, figures/pdp_twoway.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plot_style
plot_style.apply()
from config import RESULTS_DIR, MODELS_DIR, FIGURES_DIR, TARGETS

# Ordered fallback lists for each intended PDP concept.
ONE_WAY_FEATURE_GROUPS = [
    ["FEAT_AGE_YEARS", "COMP_AGE_CLIMATE"],
    "LAYER_THICKNESS_AC_MM",
    ["CLIM_FREEZE_INDEX", "CLIM_FREEZE_THAW_WINTER"],
    "UB_RESILIENT_MODULUS",
    ["CLIM_TEMP_MEAN_AVG", "COMP_WET_FREEZE"],
]

TWO_WAY_PAIR_GROUPS = [
    (["CLIM_FREEZE_INDEX", "CLIM_FREEZE_THAW_WINTER"], ["LAYER_THICKNESS_AC_MM"]),
    (["FEAT_AGE_YEARS", "COMP_AGE_CLIMATE"], ["CLIM_FREEZE_INDEX", "COMP_WET_FREEZE"]),
]

N_GRID_1WAY = 50
N_GRID_2WAY = 30


def quantile_grid(values: np.ndarray, n: int) -> np.ndarray:
    valid = values[~np.isnan(values)]
    quantiles = np.linspace(0, 100, n)
    return np.unique(np.percentile(valid, quantiles))


def partial_dependence_1d(model, X: np.ndarray, feature_idx: int,
                           grid: np.ndarray) -> np.ndarray:
    """Average prediction over marginal distribution at each grid point."""
    X_copy = X.copy()
    means  = []
    for val in grid:
        X_copy[:, feature_idx] = val
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            means.append(model.predict(X_copy).mean())
    return np.array(means)


def partial_dependence_2d(model, X: np.ndarray, idx1: int, idx2: int,
                           grid1: np.ndarray, grid2: np.ndarray) -> np.ndarray:
    Z = np.zeros((len(grid1), len(grid2)))
    X_copy = X.copy()
    for i, v1 in enumerate(grid1):
        for j, v2 in enumerate(grid2):
            X_copy[:, idx1] = v1
            X_copy[:, idx2] = v2
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                Z[i, j] = model.predict(X_copy).mean()
    return Z


def unscale_grid(scaler, feat_name: str, grid: np.ndarray) -> np.ndarray:
    """Convert standardized grid back to original units using saved scaler."""
    feat_names = list(scaler.feature_names_in_)
    if feat_name not in feat_names:
        return grid  # compound features created post-scaling — keep scaled
    i = feat_names.index(feat_name)
    return grid * scaler.scale_[i] + scaler.mean_[i]


def main():
    target_name = "iri"
    target_col  = TARGETS[target_name]
    arch        = "xgb"
    task        = "design"

    model_f  = MODELS_DIR / f"{arch}_{task}_{target_name}.joblib"
    train_f  = RESULTS_DIR / f"{task}_{target_name}_train.parquet"
    scaler_f = MODELS_DIR / "scaler_iri.joblib"

    if not model_f.exists() or not train_f.exists():
        print("Model or training data not found — run 08_tasks.py and 10_train.py first.")
        return

    model, feature_cols = joblib.load(model_f)
    scaler   = joblib.load(scaler_f) if scaler_f.exists() else None
    train_df = pd.read_parquet(train_f)
    X_train  = train_df[feature_cols].fillna(0).values

    feat_idx = {f: i for i, f in enumerate(feature_cols)}

    def choose_feature(option):
        options = option if isinstance(option, list) else [option]
        return next((f for f in options if f in feat_idx), None)

    # x-axis labels with original units (unscaled); compound features use standardized
    FEAT_LABELS = {
        "FEAT_AGE_YEARS":          "Pavement Age (years)",
        "COMP_AGE_CLIMATE":        "Age × Freeze Index Compound\n(standardized)",
        "LAYER_THICKNESS_AC_MM":   "AC Layer Thickness (mm)",
        "CLIM_FREEZE_INDEX":       "Freeze Index (°C·days)",
        "CLIM_FREEZE_THAW_WINTER": "Winter Freeze-Thaw Cycles (per year)",
        "CLIM_TEMP_MEAN_AVG":      "Mean Temperature (°C)",
        "UB_RESILIENT_MODULUS":    "Subgrade Resilient Modulus (MPa)",
        "COMP_WET_FREEZE":         "Wet-Freeze Compound\n(standardized, FT × Precipitation)",
    }

    PANEL_LETTERS = ["(a)", "(b)", "(c)", "(d)", "(e)"]

    # ── One-way PDPs ──────────────────────────────────────────────────────────
    valid_feats = [f for f in (choose_feature(g) for g in ONE_WAY_FEATURE_GROUPS) if f]
    n    = len(valid_feats)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5 * nrows))
    axes = axes.flatten() if n > 1 else [axes]

    for i, feat in enumerate(valid_feats):
        idx      = feat_idx[feat]
        col_vals = train_df[feat].values if feat in train_df.columns else X_train[:, idx]
        grid_sc  = quantile_grid(col_vals, N_GRID_1WAY)
        pdp      = partial_dependence_1d(model, X_train, idx, grid_sc)

        # Unscale to original units if possible
        grid_plot = unscale_grid(scaler, feat, grid_sc) if scaler else grid_sc

        ax = axes[i]
        ax.plot(grid_plot, pdp, color="#4e79a7", linewidth=2.2)
        ax.fill_between(grid_plot, pdp.min(), pdp, alpha=0.15, color="#4e79a7")
        xlabel = FEAT_LABELS.get(feat, feat.replace("_", " "))
        ax.set_xlabel(xlabel.split("\n")[0], fontsize=9)
        ax.set_ylabel("Mean predicted IRI (m/km)", fontsize=9)
        short_title = xlabel.split("\n")[0]
        ax.set_title(f"{PANEL_LETTERS[i]}  {short_title}", fontsize=10, loc="left")
        ax.grid(alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("One-way Partial Dependence — XGBoost IRI Design Model\n"
                 "(computed on training set, 50 quantile-spaced grid points; "
                 "x-axes in original units where measurable)",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pdp_oneway.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved → figures/pdp_oneway.png")

    # ── Two-way PDPs ──────────────────────────────────────────────────────────
    valid_pairs = []
    for g1, g2 in TWO_WAY_PAIR_GROUPS:
        f1, f2 = choose_feature(g1), choose_feature(g2)
        if f1 and f2:
            valid_pairs.append((f1, f2))
    if not valid_pairs:
        print("  Two-way PDP: no valid feature pairs in model — skip")
    else:
        fig, axes = plt.subplots(1, len(valid_pairs),
                                 figsize=(7.5 * len(valid_pairs), 6.5))
        if len(valid_pairs) == 1:
            axes = [axes]

        two_letters = ["(A)", "(B)"]
        for k, (ax, (feat1, feat2)) in enumerate(zip(axes, valid_pairs)):
            idx1  = feat_idx[feat1]
            idx2  = feat_idx[feat2]
            col1  = train_df[feat1].values if feat1 in train_df.columns else X_train[:, idx1]
            col2  = train_df[feat2].values if feat2 in train_df.columns else X_train[:, idx2]
            grid1_sc = quantile_grid(col1, N_GRID_2WAY)
            grid2_sc = quantile_grid(col2, N_GRID_2WAY)

            # Unscale for axis labels
            g1 = unscale_grid(scaler, feat1, grid1_sc) if scaler else grid1_sc
            g2 = unscale_grid(scaler, feat2, grid2_sc) if scaler else grid2_sc

            Z  = partial_dependence_2d(model, X_train, idx1, idx2, grid1_sc, grid2_sc)
            im = ax.contourf(g2, g1, Z, levels=20, cmap="RdYlBu_r")
            cb = plt.colorbar(im, ax=ax)
            cb.set_label("Mean predicted IRI (m/km)", fontsize=9)

            xlabel = FEAT_LABELS.get(feat2, feat2.replace("_", " ")).split("\n")[0]
            ylabel = FEAT_LABELS.get(feat1, feat1.replace("_", " ")).split("\n")[0]
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(f"{two_letters[k]}  {ylabel}\n× {xlabel}", fontsize=10, loc="left")

        fig.suptitle("Two-way Partial Dependence — XGBoost IRI Design Model\n"
                     "(30 × 30 quantile grid, computed on training set; "
                     "blue = lower IRI, red = higher IRI)",
                     fontsize=11)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "pdp_twoway.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved → figures/pdp_twoway.png")

    print("\n[14] PDP analysis complete.\n")


if __name__ == "__main__":
    main()
