"""
15_residuals.py — Residual diagnostic plots for test evaluation and LOO results.
Methodology §3.11.4: four diagnostic plots per evaluation context.
  1. Residuals vs. predicted value
  2. Residuals vs. observation year
  3. Residuals by region (box plots)
  4. Residuals vs. freeze index
Outputs: figures/residual_diagnostics.png
         figures/loo_scatter_4panel.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plot_style
plot_style.apply()
from config import RESULTS_DIR, FIGURES_DIR, TARGETS, COL_REGION, REGIONS


def four_residual_panels(pred_df: pd.DataFrame, target_col: str,
                          title: str, out_path: Path) -> None:
    residuals = pred_df["residual"].values
    predicted = pred_df["predicted"].values

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=13)

    # Panel 1: Residuals vs. predicted
    ax = axes[0, 0]
    ax.scatter(predicted, residuals, alpha=0.5, s=15, color="steelblue")
    ax.axhline(0, color="red", linewidth=1)
    ax.set_xlabel("Predicted IRI")
    ax.set_ylabel("Residual (Actual − Predicted)")
    ax.set_title("Residuals vs. Predicted")
    ax.grid(alpha=0.3)

    # Panel 2: Residuals vs. year
    ax = axes[0, 1]
    if "OBSERVATION_DATE" in pred_df.columns:
        years = pd.to_datetime(pred_df["OBSERVATION_DATE"], errors="coerce").dt.year
        ax.scatter(years, residuals, alpha=0.5, s=15, color="darkorange")
        ax.axhline(0, color="red", linewidth=1)
        ax.set_xlabel("Observation Year")
        ax.set_ylabel("Residual")
        ax.set_title("Residuals vs. Year (temporal drift check)")
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Date not available", ha="center", va="center")
        ax.set_visible(False)

    # Panel 3: Residuals by region (box plots)
    ax = axes[1, 0]
    region_data = []
    region_labels = []
    for region in REGIONS:
        mask = pred_df[COL_REGION] == region
        if mask.sum() > 0:
            region_data.append(residuals[mask.values])
            region_labels.append(region)
    if region_data:
        bp = ax.boxplot(region_data, labels=region_labels, patch_artist=True,
                        medianprops={"color": "red", "linewidth": 2})
        colors = ["#4e79a7", "#f28e2b", "#76b7b2", "#e15759"]
        for patch, color in zip(bp["boxes"], colors[:len(region_data)]):
            patch.set_facecolor(color)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_ylabel("Residual")
        ax.set_title("Residuals by Region")
        ax.grid(alpha=0.3, axis="y")

    # Panel 4: Residuals vs. freeze-thaw proxy (CLIM_FREEZE_INDEX excluded by step 07)
    ax = axes[1, 1]
    proxy_col = "CLIM_FREEZE_THAW_WINTER"
    fi_col    = "CLIM_FREEZE_INDEX"
    if fi_col in pred_df.columns:
        fi = pred_df[fi_col].values
        ax.scatter(fi, residuals, alpha=0.5, s=15, color="purple")
        ax.axhline(0, color="red", linewidth=1)
        ax.set_xlabel("Freeze Index (°C·days)")
        ax.set_ylabel("Residual")
        ax.set_title("Residuals vs. Freeze Index")
        ax.grid(alpha=0.3)
    elif proxy_col in pred_df.columns:
        ftz = pred_df[proxy_col].values
        mask = ~np.isnan(ftz)
        ax.scatter(ftz[mask], residuals[mask], alpha=0.5, s=15, color="purple")
        ax.axhline(0, color="red", linewidth=1)
        ax.set_xlabel("Winter Freeze-Thaw Cycles (standardized)")
        ax.set_ylabel("Residual")
        ax.set_title("Residuals vs. Freeze-Thaw Proxy\n"
                     "(CLIM_FREEZE_THAW_WINTER; CLIM_FREEZE_INDEX excluded step 07)")
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5,
                "CLIM_FREEZE_INDEX excluded by\ncollinearity reduction (step 07).\n"
                "See CLIM_FREEZE_THAW_WINTER\nPDPs in Figure 11.",
                ha="center", va="center", fontsize=9, color="dimgray")
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def loo_scatter_4panel(out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    axes = axes.flatten()
    fig.suptitle("LOO Validation — Predicted vs. Actual IRI by Withheld Region",
                 fontsize=13)

    for ax, region in zip(axes, REGIONS):
        loo_f = RESULTS_DIR / f"loo_{region.lower()}_xgb.csv"
        if not loo_f.exists():
            ax.text(0.5, 0.5, f"No LOO data\n{region}", ha="center", va="center")
            continue
        df = pd.read_csv(loo_f)
        y_true = df[TARGETS["iri"]].values
        y_pred = df["predicted"].values

        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        yt, yp = y_true[mask], y_pred[mask]

        from sklearn.metrics import r2_score
        r2 = r2_score(yt, yp) if len(yt) >= 2 else np.nan

        lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        ax.scatter(yt, yp, alpha=0.6, s=20, color="#4e79a7")
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.2, label="1:1 line")
        ax.set_xlabel("Actual IRI (m/km)")
        ax.set_ylabel("Predicted IRI (m/km)")
        ax.set_title(f"Withheld: {region}  (R²={r2:+.3f})")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    target_name = "iri"
    target_col  = TARGETS[target_name]

    # Standard test residual diagnostics (XGBoost design model)
    pred_f = RESULTS_DIR / f"test_predictions_xgb_design_{target_name}.parquet"
    if pred_f.exists():
        pred_df = pd.read_parquet(pred_f)

        # Merge freeze-related columns back if not already in predictions
        test_src = RESULTS_DIR / f"design_{target_name}_test.parquet"
        if test_src.exists():
            import pyarrow.parquet as pq
            available = pq.read_schema(test_src).names
            cols = ["section_key", "OBSERVATION_DATE"]
            for extra in ["CLIM_FREEZE_INDEX", "CLIM_FREEZE_THAW_WINTER"]:
                if extra in available and extra not in pred_df.columns:
                    cols.append(extra)
            if len(cols) > 2:
                fi_df = pd.read_parquet(test_src, columns=cols)
                pred_df = pred_df.merge(fi_df, on=["section_key", "OBSERVATION_DATE"], how="left")

        four_residual_panels(
            pred_df, target_col,
            "Residual Diagnostics — XGBoost Design Model (IRI, Test Set)",
            FIGURES_DIR / "residual_diagnostics.png",
        )
        print("Saved → figures/residual_diagnostics.png")
    else:
        print("Test predictions not found — run 11_evaluate.py first.")

    # LOO 4-panel scatter
    loo_scatter_4panel(FIGURES_DIR / "loo_scatter_4panel.png")
    print("Saved → figures/loo_scatter_4panel.png")

    # Ontario sensitivity plot
    ont_f = RESULTS_DIR / "ontario_section_loo.csv"
    if ont_f.exists():
        ont_df = pd.read_csv(ont_f)
        xgb_ont = ont_df[ont_df["arch"] == "xgb"].dropna(subset=["mean_fi", "R2"])
        if len(xgb_ont):
            fig, ax = plt.subplots(figsize=(9, 6))
            colors = {"moderate": "#4e79a7", "high": "#f28e2b", "extreme": "#e15759"}

            # Smart annotation offsets to avoid overlap for co-located points
            offsets = {}
            for _, row in xgb_ont.iterrows():
                key = round(row["mean_fi"], 0)
                offsets[key] = offsets.get(key, 0) + 1

            used = {}
            for _, row in xgb_ont.iterrows():
                c   = colors.get(row["fi_group"], "gray")
                key = round(row["mean_fi"], 0)
                ax.scatter(row["mean_fi"], row["R2"], color=c, s=130,
                           zorder=4, edgecolors="white", linewidths=1.2)
                # Alternate annotation side for clustered points
                used[key] = used.get(key, 0) + 1
                side = 1 if used[key] % 2 == 1 else -1
                xt = 8 * side
                yt = 5 if row["R2"] > -1.5 else -12
                ax.annotate(f"#{int(row['shrp_id'])}", (row["mean_fi"], row["R2"]),
                            fontsize=9, textcoords="offset points", xytext=(xt, yt),
                            ha="left" if side > 0 else "right")

            for label, color in colors.items():
                ax.scatter([], [], color=color, label=f"{label.capitalize()} freeze", s=80)
            ax.axhline(0, color="red", linewidth=1.2, linestyle="--", label="R²=0 (no skill)")
            ax.set_xlabel("Mean Annual Freeze Index (°C·days)", fontsize=11)
            ax.set_ylabel("Per-section LOO R²  (exploratory, no CI)", fontsize=11)
            ax.set_title("Ontario Section-level LOO R² vs. Freeze Index\n"
                         "(7 leave-one-section-out sub-iterations, XGBoost)", fontsize=12)
            ax.legend(loc="lower right", fontsize=9)
            fi_vals = xgb_ont["mean_fi"].values
            r2_vals = xgb_ont["R2"].values
            pad = (fi_vals.max() - fi_vals.min()) * 0.12
            ax.set_xlim(fi_vals.min() - pad, fi_vals.max() + pad + 40)
            y_min = r2_vals.min() - 0.4
            y_max = r2_vals.max() + 0.25
            ax.set_ylim(min(y_min, -0.6), max(y_max, 0.4))
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / "ontario_sensitivity_scatter.png",
                        dpi=150, bbox_inches="tight")
            plt.close()
            print("Saved → figures/ontario_sensitivity_scatter.png")

    print("\n[15] Residual diagnostics complete.\n")


if __name__ == "__main__":
    main()
