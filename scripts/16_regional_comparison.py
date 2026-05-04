"""
16_regional_comparison.py — Regional climate and traffic comparison figures.
Produces four figures needed to compare model behaviour across the climate gradient:
  1. figures/regional_climate_traffic_profile.png
       2×2 panel: freeze index, mean IRI by region, AADTT traffic, IRI vs age by region
  2. figures/shap_category_by_region.png
       Stacked-bar: SHAP importance split into Climate / Traffic / Structure per region
  3. figures/loo_vs_climate_gradient.png
       LOO R² (XGB) vs mean freeze index — the central "limits" figure
  4. figures/regional_iri_trajectories.png
       IRI vs pavement age with LOWESS trend lines, one trace per region
Outputs written to figures/. Raw data read from data/ and results/.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.stats import linregress
import plot_style
plot_style.apply()
from config import RESULTS_DIR, FIGURES_DIR, TARGETS, COL_REGION, REGIONS, RAW_CSV


# ── Colour palette (consistent with earlier figures) ─────────────────────────
REGION_COLORS = {
    "Arizona": "#4e79a7",
    "Georgia": "#f28e2b",
    "Ohio":    "#76b7b2",
    "Ontario": "#e15759",
}
CAT_COLORS = {
    "Structure": "#4e79a7",
    "Traffic":   "#f28e2b",
    "Climate":   "#76b7b2",
}

MEAN_FI = {"Arizona": 5.9, "Georgia": 10.9, "Ohio": 316.6, "Ontario": 834.9}
MEAN_TEMP = {"Arizona": 18.5, "Georgia": 17.8, "Ohio": 10.6, "Ontario": 6.4}
MEAN_AADTT = {"Arizona": 1354.7, "Georgia": 851.3, "Ohio": 1566.3, "Ontario": 970.2}
MEAN_ESAL12 = {"Arizona": 0.5, "Georgia": 0.1, "Ohio": 0.5, "Ontario": 1.3}


def categorize_feature(feat: str) -> str:
    if (feat.startswith("CLIM") or feat in
            {"COMP_AGE_CLIMATE", "COMP_WET_FREEZE", "COMP_MAT_FREEZE_SUSC"}):
        return "Climate"
    if (feat.startswith("TRF") or "ESAL" in feat or "AADTT" in feat):
        return "Traffic"
    return "Structure"


# ── Figure 1: Regional climate & traffic profile ──────────────────────────────
def plot_climate_traffic_profile(raw_df: pd.DataFrame, out_path: Path) -> None:
    fig = plt.figure(figsize=(16, 12))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

    region_order = REGIONS

    # Panel A — Freeze index (log scale)
    ax = fig.add_subplot(gs[0, 0])
    fi_vals = [MEAN_FI[r] for r in region_order]
    bars = ax.bar(region_order, fi_vals,
                  color=[REGION_COLORS[r] for r in region_order], edgecolor="white")
    ax.set_yscale("log")
    ax.set_ylabel("Mean Annual Freeze Index (°C·days, log scale)")
    ax.set_title("(A) Climate Severity — Freeze Index")
    for bar, v in zip(bars, fi_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v * 1.08,
                f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Panel B — Mean IRI ± 1 std by region
    ax = fig.add_subplot(gs[0, 1])
    iri_means, iri_stds = [], []
    for r in region_order:
        vals = raw_df[raw_df[COL_REGION] == r]["IRI_MRI"].dropna()
        iri_means.append(vals.mean())
        iri_stds.append(vals.std())
    ax.bar(region_order, iri_means, yerr=iri_stds, capsize=5,
           color=[REGION_COLORS[r] for r in region_order],
           edgecolor="white", error_kw={"ecolor": "gray", "linewidth": 1.2})
    ax.set_ylabel("IRI (m/km)")
    ax.set_title("(B) Mean IRI ± 1 SD by Region\n(raw observations, all years)")
    ax.axhline(1.5, color="red", linewidth=1, linestyle="--", alpha=0.6, label="Threshold 1.5")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Panel C — AADTT and ESAL class 12 side by side
    ax  = fig.add_subplot(gs[1, 0])
    ax2 = ax.twinx()
    x   = np.arange(len(region_order))
    w   = 0.35
    aadtt_vals  = [MEAN_AADTT[r]  for r in region_order]
    esal12_vals = [MEAN_ESAL12[r] for r in region_order]
    b1 = ax.bar(x - w/2, aadtt_vals, width=w, label="AADTT (left)",
                color=[REGION_COLORS[r] for r in region_order], edgecolor="white")
    b2 = ax2.bar(x + w/2, esal12_vals, width=w, label="ESAL class 12 (right)",
                 color=[REGION_COLORS[r] for r in region_order], edgecolor="white",
                 alpha=0.55, hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels(region_order)
    ax.set_ylabel("AADTT (trucks/lane/year)")
    ax2.set_ylabel("Mean ESAL per class-12 vehicle")
    ax.set_title("(C) Traffic Loading by Region")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # Panel D — IRI vs pavement age, coloured by region
    ax = fig.add_subplot(gs[1, 1])
    for r in region_order:
        sub = raw_df[raw_df[COL_REGION] == r][["AGE_YEARS", "IRI_MRI"]].dropna()
        sub = sub[(sub["AGE_YEARS"] >= 0) & (sub["AGE_YEARS"] <= 35)]
        ax.scatter(sub["AGE_YEARS"], sub["IRI_MRI"],
                   alpha=0.18, s=8, color=REGION_COLORS[r])
        # LOWESS-style binned mean
        bins = np.arange(0, 36, 3)
        sub["bin"] = pd.cut(sub["AGE_YEARS"], bins)
        bm = sub.groupby("bin", observed=True)["IRI_MRI"].mean()
        bin_centers = [b.mid for b in bm.index]
        ax.plot(bin_centers, bm.values, color=REGION_COLORS[r], linewidth=2.2,
                label=r)
    ax.set_xlabel("Pavement Age (years)")
    ax.set_ylabel("IRI (m/km)")
    ax.set_title("(D) IRI vs Pavement Age by Region\n(binned means ± scatter)")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(0, 4)
    ax.grid(alpha=0.3)

    fig.suptitle("Regional Climate, Traffic, and Deterioration Profile", fontsize=14, y=1.01)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path.relative_to(out_path.parent.parent)}")


# ── Figure 2: SHAP importance by feature category per region ─────────────────
def plot_shap_category_by_region(out_path: Path) -> None:
    sr = pd.read_csv(RESULTS_DIR / "shap_regional.csv", index_col=0)
    sr["category"] = sr.index.map(categorize_feature)
    cat_sum = sr.groupby("category")[REGIONS].sum()

    # Normalise to % of total SHAP per region
    pct = cat_sum.div(cat_sum.sum(axis=0), axis=1) * 100

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # Left — absolute mean |SHAP| per category per region
    ax = axes[0]
    x  = np.arange(len(REGIONS))
    w  = 0.22
    for i, cat in enumerate(["Structure", "Traffic", "Climate"]):
        vals = [cat_sum.loc[cat, r] if cat in cat_sum.index else 0 for r in REGIONS]
        ax.bar(x + i * w, vals, width=w, label=cat,
               color=CAT_COLORS[cat], edgecolor="white")
    ax.set_xticks(x + w)
    ax.set_xticklabels(REGIONS)
    ax.set_ylabel("Sum of mean |SHAP| values")
    ax.set_title("(A) Absolute SHAP by Feature Category\n(XGBoost IRI design model)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Right — stacked 100% bar showing proportional split
    ax = axes[1]
    bottom = np.zeros(len(REGIONS))
    for cat in ["Structure", "Traffic", "Climate"]:
        vals = pct.loc[cat].values if cat in pct.index else np.zeros(len(REGIONS))
        ax.bar(REGIONS, vals, bottom=bottom, label=cat,
               color=CAT_COLORS[cat], edgecolor="white")
        for j, (v, b) in enumerate(zip(vals, bottom)):
            if v > 4:
                ax.text(j, b + v / 2, f"{v:.0f}%",
                        ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        bottom += vals
    ax.set_ylabel("% of total SHAP importance")
    ax.set_title("(B) Proportional SHAP Split by Region\n(XGBoost IRI design model)")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(0, 115)
    ax.grid(axis="y", alpha=0.3)

    # Annotate with region freeze index (positioned above 100% bar with room to breathe)
    for i, r in enumerate(REGIONS):
        ax.text(i, 104, f"FI={MEAN_FI[r]:.0f}", ha="center", fontsize=8,
                color="gray", style="italic")

    fig.suptitle("SHAP Feature Importance by Category across Climate Regions", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path.relative_to(out_path.parent.parent)}")


# ── Figure 3: LOO R² vs climate gradient ─────────────────────────────────────
def plot_loo_vs_climate(out_path: Path) -> None:
    loo = pd.read_csv(RESULTS_DIR / "loo_summary.csv")
    xgb = loo[loo["arch"] == "xgb"].copy()
    rf  = loo[loo["arch"] == "rf"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, df, arch_label in [(axes[0], xgb, "XGBoost"), (axes[1], rf, "Random Forest")]:
        fi_vals = [MEAN_FI[r] for r in df["withheld_region"]]
        r2_vals = df["R2"].values

        ax.axhline(0, color="red", linewidth=1.2, linestyle="--", alpha=0.7, label="R²=0 (no skill)")
        for fi, r2, region in zip(fi_vals, r2_vals, df["withheld_region"]):
            ax.scatter(fi, r2, s=200, color=REGION_COLORS[region],
                       zorder=5, edgecolors="white", linewidths=1.5)
            ax.annotate(region, (fi, r2), fontsize=10, fontweight="bold",
                        textcoords="offset points", xytext=(8, 4))

        # Shade interpolation vs extrapolation zones
        ax.axvspan(0, 350, alpha=0.06, color="green", label="Interpolation zone\n(FI in training range)")
        ax.axvspan(350, 1700, alpha=0.06, color="red", label="Extrapolation zone\n(FI outside training)")

        ax.set_xscale("log")
        ax.set_xlabel("Mean Freeze Index of Withheld Region (°C·days, log scale)", fontsize=10)
        ax.set_ylabel("LOO R² (IRI design model)", fontsize=10)
        ax.set_title(f"({['A','B'][axes.tolist().index(ax)]}) {arch_label} — LOO R² vs Freeze Index")
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(alpha=0.3)
        ax.set_ylim(-2.0, 0.4)

    fig.suptitle("LOO Generalisation Failure vs Climate Gradient\n"
                 "(Withheld region R² — lower = worse transfer)", fontsize=13)
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    # Shared note placed after tight_layout so bbox_inches="tight" captures it
    fig.text(0.5, 0.01,
             "Georgia's extreme underperformance (R²≈−0.8 to −1.1) reflects wet-subtropical conditions "
             "absent from the training distribution despite a low FI.",
             ha="center", fontsize=9, color="dimgray", style="italic",
             wrap=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path.relative_to(out_path.parent.parent)}")


# ── Figure 4: Top-10 features per region side by side ────────────────────────
def plot_regional_top_features(out_path: Path) -> None:
    sr = pd.read_csv(RESULTS_DIR / "shap_regional.csv", index_col=0)
    sr["category"] = sr.index.map(categorize_feature)

    fig, axes = plt.subplots(1, 4, figsize=(18, 7), sharey=False)
    fig.suptitle("Top-10 Features by SHAP Importance per Region\n"
                 "(XGBoost IRI design model — test set)", fontsize=13)

    for ax, region in zip(axes, REGIONS):
        top = sr[region].nlargest(10).sort_values()
        colors = [CAT_COLORS.get(sr.loc[f, "category"], "gray") for f in top.index]
        top.plot(kind="barh", ax=ax, color=colors)
        ax.set_title(f"{region}\n(FI={MEAN_FI[region]:.0f})", fontsize=10)
        ax.set_xlabel("Mean |SHAP|", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="x", alpha=0.3)
        # Clean and truncate feature name labels to prevent overflow
        def _clean_label(s):
            s = s.replace("_", " ").replace("TRF REP REP ", "").replace("TRF AADTT ", "AADTT ")
            s = s.replace("TRF CMLTV VOL ", "CumVol ").replace("COMP ", "").replace("CLIM ", "")
            return (s[:20] + "…") if len(s) > 22 else s
        labels = [_clean_label(t.get_text()) for t in ax.get_yticklabels()]
        ax.set_yticklabels(labels)

    # Legend for categories — below figure with extra padding
    patches = [mpatches.Patch(color=CAT_COLORS[c], label=c) for c in CAT_COLORS]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path.relative_to(out_path.parent.parent)}")


def main():
    raw_df = pd.read_csv(RAW_CSV, low_memory=False)
    raw_df["OBSERVATION_DATE"] = pd.to_datetime(raw_df["OBSERVATION_DATE"], errors="coerce")
    raw_df["CN_ASSIGN_DATE"]   = pd.to_datetime(raw_df["CN_ASSIGN_DATE"],   errors="coerce")
    raw_df["AGE_YEARS"] = (
        (raw_df["OBSERVATION_DATE"] - raw_df["CN_ASSIGN_DATE"]).dt.days / 365.25
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plot_climate_traffic_profile(
            raw_df, FIGURES_DIR / "regional_climate_traffic_profile.png"
        )
        plot_shap_category_by_region(
            FIGURES_DIR / "shap_category_by_region.png"
        )
        plot_loo_vs_climate(
            FIGURES_DIR / "loo_vs_climate_gradient.png"
        )
        plot_regional_top_features(
            FIGURES_DIR / "regional_top_features.png"
        )

    print("\n[16] Regional comparison figures complete.\n")


if __name__ == "__main__":
    main()
