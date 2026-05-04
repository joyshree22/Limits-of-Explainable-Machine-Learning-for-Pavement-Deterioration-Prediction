"""
17_waterfall.py — SHAP waterfall plots: one representative section per region.
§3.12.1: decompose prediction for one observation per climate zone into additive
feature contributions from the dataset baseline E[f(X)].
Representative section chosen as the observation closest to the regional median IRI.
Output: figures/waterfall_4panel.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import warnings
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plot_style
plot_style.apply()
from config import RESULTS_DIR, MODELS_DIR, FIGURES_DIR, TARGETS, COL_REGION, REGIONS

REGION_COLORS = {
    "Arizona": "#4e79a7",
    "Georgia": "#f28e2b",
    "Ohio":    "#59a14f",
    "Ontario": "#e15759",
}

MEAN_FI = {"Arizona": 5.9, "Georgia": 10.9, "Ohio": 316.6, "Ontario": 834.9}


def pick_representative(df, region, target_col):
    """Pick the observation from this region closest to the regional median IRI."""
    sub = df[df[COL_REGION] == region].copy()
    if len(sub) == 0:
        return None
    med = sub[target_col].median()
    idx = (sub[target_col] - med).abs().idxmin()
    return sub.loc[[idx]]


def waterfall_panel(ax, shap_vals, feature_names, base_value,
                    pred_value, actual_value, title, top_n=12, color="#4e79a7"):
    """Draw a horizontal waterfall chart on ax."""
    sv    = np.array(shap_vals)
    names = list(feature_names)

    # Select top_n by |SHAP|
    order  = np.argsort(np.abs(sv))[::-1][:top_n]
    sv_top = sv[order]
    nm_top = [names[i] for i in order]

    # Sort ascending for display (bottom = smallest contribution)
    disp_order = np.argsort(sv_top)
    sv_disp = sv_top[disp_order]
    nm_disp = [nm_top[i] for i in disp_order]

    colors = ["#e15759" if v > 0 else "#4e79a7" for v in sv_disp]
    y_pos  = np.arange(len(sv_disp))

    ax.barh(y_pos, sv_disp, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)

    # Clean feature names for display
    RENAME = {
        "TRF_REP_REP_ESAL_PER_VEH_CLASS_12": "ESAL / Class-12 Vehicle",
        "TRF_REP_REP_ESAL_PER_VEH_CLASS_10": "ESAL / Class-10 Vehicle",
        "TRF_REP_REP_ESAL_PER_VEH_CLASS_9":  "ESAL / Class-9 Vehicle",
        "TRF_AADTT_VEH_CLASS_4_TREND":        "AADTT Class-4 Trend",
        "TRF_AADTT_VEH_CLASS_5_TREND":        "AADTT Class-5 Trend",
        "TRF_AADTT_VEH_CLASS_8_TREND":        "AADTT Class-8 Trend",
        "TRF_CMLTV_VOL_VEH_CLASS_9_TREND":    "Cumul. Vol. Class-9",
        "TRF_ALDF_MEPDG_LG05":                "Axle Load Factor (MEPDG)",
        "L05B_REPR_THICKNESS":                "Base Layer Repr. Thickness",
        "L05B_MATL_CODE":                     "Base Layer Material Code",
        "L05B_INV_NO_LAYER_ND":               "Base Layer Count",
        "LAYER_COUNT_AC":                     "AC Layer Count",
        "LAYER_THICKNESS_AC_MM":              "AC Layer Thickness (mm)",
        "LAYER_SURFACE_MATL_CODE":            "Surface Material Code",
        "AC_ASPHALT_CONTENT_MEAN":            "AC Asphalt Content",
        "AC_DESCRIPTION":                     "AC Mix Description",
        "AC_BSG":                             "AC Bulk Sp. Gravity",
        "AC_IDT_POISON_LG00":                 "AC Tensile Str. Ratio",
        "UB_ONE_PASSING":                     "Base Passing #200",
        "CLIM_LONGITUDE":                     "Longitude",
        "CLIM_FREEZE_THAW_WINTER":            "Winter Freeze-Thaw Cycles",
        "CLIM_CLOUD_COVER_AVG":               "Mean Cloud Cover",
        "COMP_AGE_CLIMATE":                   "Age × Freeze Index",
        "COMP_STRUCT_ADEQUACY":               "Structural Adequacy",
        "COMP_WET_FREEZE":                    "Wet-Freeze Compound",
        "ESAL_PER_VEH_CLASS_12":              "ESAL / Class-12",
        "ESAL_PER_VEH_CLASS_10":              "ESAL / Class-10",
    }
    clean = []
    for n in nm_disp:
        if n in RENAME:
            clean.append(RENAME[n])
        else:
            s = n.replace("_", " ")
            s = s.replace("TRF REP REP ", "").replace("TRF AADTT ", "AADTT ")
            s = s.replace("TRF CMLTV VOL ", "Cumul.Vol. ")
            s = s.replace("COMP ", "").replace("CLIM ", "")
            if len(s) > 32:
                s = s[:30] + "…"
            clean.append(s)
    ax.set_yticklabels(clean, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)

    # Baseline and prediction annotations
    ax.set_xlabel("SHAP contribution (Δ IRI from baseline)", fontsize=9)
    ax.set_title(
        f"{title} | FI={MEAN_FI.get(title.split()[0], '?')}\n"
        f"Actual={actual_value:.3f}  Pred={pred_value:.3f}  Base={base_value:.3f} m/km",
        fontsize=9.5
    )

    # Positive/negative legend patches
    pos_p = mpatches.Patch(color="#e15759", label="Increases IRI")
    neg_p = mpatches.Patch(color="#4e79a7", label="Decreases IRI")
    ax.legend(handles=[pos_p, neg_p], fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)


def main():
    target_name = "iri"
    target_col  = TARGETS[target_name]
    arch        = "xgb"
    task        = "design"

    model_f  = MODELS_DIR / f"{arch}_{task}_{target_name}.joblib"
    test_f   = RESULTS_DIR / f"{task}_{target_name}_test.parquet"

    if not model_f.exists() or not test_f.exists():
        print("Model or test data not found — run steps 08 and 10 first.")
        return

    model, feature_cols = joblib.load(model_f)
    test_df = pd.read_parquet(test_f)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer   = shap.TreeExplainer(model)
        X_test      = test_df[feature_cols].fillna(0)
        shap_values = explainer.shap_values(X_test.values)
        base_value  = float(explainer.expected_value)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    for ax, region in zip(axes, REGIONS):
        rep = pick_representative(test_df, region, target_col)
        if rep is None:
            ax.text(0.5, 0.5, f"No test data\n{region}", ha="center", va="center")
            continue

        row_idx  = test_df.index.get_loc(rep.index[0])
        sv_row   = shap_values[row_idx]
        pred     = float(model.predict(X_test.values[[row_idx]])[0])
        actual   = float(rep[target_col].values[0])

        waterfall_panel(ax, sv_row, feature_cols, base_value,
                        pred, actual, f"{region} (representative section)", top_n=12)

    fig.suptitle(
        "SHAP Waterfall Decomposition — XGBoost IRI Design Model\n"
        "One representative section per climate region "
        "(closest to regional median IRI, top 12 contributing features)",
        fontsize=13
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "waterfall_4panel.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved → figures/waterfall_4panel.png")
    print("\n[17] Waterfall plots complete.\n")


if __name__ == "__main__":
    main()
