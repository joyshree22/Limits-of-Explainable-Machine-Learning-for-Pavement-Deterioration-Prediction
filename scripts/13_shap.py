"""
13_shap.py — SHAP TreeExplainer for XGBoost design models.
Methodology §3.12.1–3.12.2:
  - Exact Shapley values via TreeExplainer (not KernelExplainer).
  - Global importance: mean(|SHAP|) per feature.
  - Beeswarm plot (Figure 2), bar chart (Figure 3).
  - Regional SHAP stratification (Figure 7).
  - Cross-model consistency check: spearmanr(xgb_imp, rf_imp) directly on raw vectors.
    Pre-specified threshold ρ > 0.75; expected ρ ≈ 0.594.
Outputs: results/shap_global.csv, results/shap_regional.csv, results/shap_consistency.csv
         figures/beeswarm_iri_design.png, figures/shap_bar_iri_design.png
         figures/regional_shap.png
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
import plot_style
plot_style.apply()
from scipy.stats import spearmanr
from config import (
    RESULTS_DIR, MODELS_DIR, FIGURES_DIR, TARGETS,
    COL_REGION, REGIONS, SHAP_CONSISTENCY_THRESHOLD,
)

META_COLS = {
    "section_key", "STATE_CODE_EXP", "SHRP_ID", "PAVEMENT_FAMILY",
    "OBSERVATION_DATE", "CN_ASSIGN_DATE", "AGE_YEARS", "split",
}


def compute_shap_importance(model, X: np.ndarray, feature_cols: list) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
    importance = np.abs(shap_values).mean(axis=0)
    return pd.Series(importance, index=feature_cols), shap_values, explainer


FEAT_CLEAN = {
    "TRF_REP_REP_ESAL_PER_VEH_CLASS_12": "ESAL per Class-12 Vehicle",
    "TRF_REP_REP_ESAL_PER_VEH_CLASS_10": "ESAL per Class-10 Vehicle",
    "TRF_REP_REP_ESAL_PER_VEH_CLASS_9":  "ESAL per Class-9 Vehicle",
    "TRF_AADTT_VEH_CLASS_4_TREND":       "AADTT Class-4 Trend",
    "TRF_AADTT_VEH_CLASS_5_TREND":       "AADTT Class-5 Trend",
    "TRF_AADTT_VEH_CLASS_8_TREND":       "AADTT Class-8 Trend",
    "TRF_CMLTV_VOL_VEH_CLASS_9_TREND":   "Cumul. Vol. Class-9 Trend",
    "TRF_ALDF_MEPDG_LG05":               "Traffic Axle Load Factor (MEPDG)",
    "L05B_REPR_THICKNESS":               "Base Layer Representative Thickness",
    "L05B_MATL_CODE":                    "Base Layer Material Code",
    "LAYER_COUNT_AC":                    "AC Layer Count",
    "LAYER_THICKNESS_AC_MM":             "AC Layer Thickness (mm)",
    "LAYER_SURFACE_MATL_CODE":           "Surface Layer Material Code",
    "AC_ASPHALT_CONTENT_MEAN":           "AC Asphalt Content Mean",
    "AC_DESCRIPTION":                    "AC Mix Description",
    "AC_BSG":                            "AC Bulk Specific Gravity",
    "AC_IDT_POISON_LG00":                "AC Tensile Strength Ratio",
    "UB_ONE_PASSING":                    "Base Layer Passing #200",
    "CLIM_LONGITUDE":                    "Longitude",
    "COMP_AGE_CLIMATE":                  "Age × Freeze Index Compound",
    "COMP_STRUCT_ADEQUACY":              "Structural Adequacy Composite",
    "COMP_WET_FREEZE":                   "Wet-Freeze Compound",
}


def clean_feat_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with human-readable column names for SHAP plots."""
    rename = {}
    for col in df.columns:
        if col in FEAT_CLEAN:
            rename[col] = FEAT_CLEAN[col]
        else:
            name = col.replace("_", " ").strip()
            # remove repetitive TRF REP prefix
            name = name.replace("TRF REP REP ", "").replace("TRF AADTT ", "AADTT ")
            name = name.replace("TRF CMLTV VOL ", "Cumul.Vol. ")
            name = name.replace("TRF ALDF MEPDG ", "MEPDG ")
            # Truncate long names
            if len(name) > 38:
                name = name[:36] + "…"
            rename[col] = name
    return df.rename(columns=rename)


def plot_beeswarm(shap_values, X_df, out_path, title="SHAP Beeswarm — XGBoost IRI Design Model"):
    X_clean = clean_feat_names(X_df)
    plt.figure(figsize=(11, 9))
    shap.summary_plot(shap_values, X_clean, show=False, max_display=20)
    plt.title(title, fontsize=12, pad=12)
    plt.xlabel("SHAP value (impact on predicted IRI, m/km)", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_bar(importance: pd.Series, out_path,
             title="Global Mean |SHAP| — XGBoost IRI Design Model"):
    top = importance.nlargest(20)
    # Clean index names
    dummy = pd.DataFrame(columns=top.index)
    clean_cols = clean_feat_names(dummy).columns
    top_clean = pd.Series(top.values, index=clean_cols)
    fig, ax = plt.subplots(figsize=(9, 7))
    top_clean[::-1].plot(kind="barh", ax=ax, color="#4e79a7", edgecolor="white")
    ax.set_xlabel("Mean |SHAP value| (m/km)", fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_regional_shap(regional_imp: pd.DataFrame, out_path):
    top_feats = regional_imp.mean(axis=1).nlargest(15).index
    sub = regional_imp.loc[top_feats]

    # Shorten feature names so legend items don't overflow
    def _shorten(n):
        n = n.replace("TRF_REP_REP_", "").replace("TRF_AADTT_", "AADTT_")
        n = n.replace("TRF_CMLTV_VOL_", "CumVol_").replace("COMP_", "")
        n = n.replace("CLIM_", "").replace("_", " ")
        return n[:22] + "…" if len(n) > 24 else n

    sub.index = [_shorten(f) for f in sub.index]
    fig, ax = plt.subplots(figsize=(12, 7))
    sub.T.plot(kind="bar", ax=ax, colormap="tab10", width=0.7)
    ax.set_title("SHAP Importance by Region — Top 15 Features\n(XGBoost IRI design model)",
                 fontsize=12)
    ax.set_xlabel("Region", fontsize=10)
    ax.set_ylabel("Mean |SHAP value| (m/km)", fontsize=10)
    ax.tick_params(axis="x", rotation=0, labelsize=10)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="y", alpha=0.3)
    # Place legend outside right edge so it never overlaps bars
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1),
              borderaxespad=0, ncol=1, frameon=True)
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    # Primary: IRI design model — main SHAP analysis
    target_name = "iri"
    target_col  = TARGETS[target_name]
    task        = "design"

    test_f = RESULTS_DIR / f"{task}_{target_name}_test.parquet"
    if not test_f.exists():
        print("Test dataset not found — run 08_tasks.py first.")
        return

    test_df = pd.read_parquet(test_f)

    global_rows     = []
    regional_dict   = {}
    xgb_importance  = None
    rf_importance   = None

    for arch in ["xgb", "rf"]:
        model_f = MODELS_DIR / f"{arch}_{task}_{target_name}.joblib"
        if not model_f.exists():
            print(f"  Model not found: {model_f.name}")
            continue

        model, feature_cols = joblib.load(model_f)
        X_test = test_df[feature_cols].fillna(0)

        importance, shap_values, explainer = compute_shap_importance(
            model, X_test.values, feature_cols
        )

        # Global importance
        for feat, imp in importance.items():
            global_rows.append({
                "arch":       arch,
                "target":     target_name,
                "task":       task,
                "feature":    feat,
                "mean_abs_shap": round(float(imp), 6),
            })

        # Store for cross-model consistency check
        if arch == "xgb":
            xgb_importance = importance
        elif arch == "rf":
            rf_importance  = importance

        # Regional SHAP stratification
        reg_imp = {}
        for region in REGIONS:
            region_mask = (test_df[COL_REGION] == region).values
            if region_mask.sum() < 2:
                continue
            region_imp = np.abs(shap_values[region_mask]).mean(axis=0)
            reg_imp[region] = pd.Series(region_imp, index=feature_cols)
        regional_dict[arch] = pd.DataFrame(reg_imp)   # features × regions

        # Figures for XGBoost only (primary architecture)
        if arch == "xgb":
            plot_beeswarm(
                shap_values, X_test,
                FIGURES_DIR / "beeswarm_iri_design.png"
            )
            plot_bar(
                importance,
                FIGURES_DIR / "shap_bar_iri_design.png"
            )
            print("  Saved → figures/beeswarm_iri_design.png")
            print("  Saved → figures/shap_bar_iri_design.png")

    # Save global importance
    pd.DataFrame(global_rows).to_csv(
        RESULTS_DIR / "shap_global.csv", index=False
    )

    # Save regional importance (XGBoost)
    if "xgb" in regional_dict and not regional_dict["xgb"].empty:
        regional_dict["xgb"].to_csv(RESULTS_DIR / "shap_regional.csv")
        plot_regional_shap(
            regional_dict["xgb"],
            FIGURES_DIR / "regional_shap.png"
        )
        print("  Saved → figures/regional_shap.png")

    # ── Cross-model SHAP consistency check (§3.12.2) ──────────────────────────
    if xgb_importance is not None and rf_importance is not None:
        # Align on common features
        common = xgb_importance.index.intersection(rf_importance.index)
        xgb_v  = xgb_importance[common].values
        rf_v   = rf_importance[common].values

        # spearmanr applied directly to raw importance vectors (not pre-ranked)
        rho, pval = spearmanr(xgb_v, rf_v)

        threshold = SHAP_CONSISTENCY_THRESHOLD
        status    = "STABLE (ρ > 0.75)" if rho > threshold else (
            f"UNSTABLE (ρ = {rho:.4f} < {threshold}) — "
            "robust claims restricted to top-ranking converging features"
        )
        print(f"\n  SHAP cross-model consistency: ρ = {rho:.4f}, p = {pval:.4f}")
        print(f"  Status: {status}")
        print(f"  Note: ρ estimate based on 7 test sections — treat as approximate.")

        pd.DataFrame([{
            "n_features":       len(common),
            "spearman_rho":     round(rho, 4),
            "p_value":          round(pval, 4),
            "threshold":        threshold,
            "status":           status,
        }]).to_csv(RESULTS_DIR / "shap_consistency.csv", index=False)
        print("  Saved → results/shap_consistency.csv")

    print("\nSaved → results/shap_global.csv")
    print("Saved → results/shap_regional.csv")
    print("\n[13] SHAP analysis complete.\n")


if __name__ == "__main__":
    main()
