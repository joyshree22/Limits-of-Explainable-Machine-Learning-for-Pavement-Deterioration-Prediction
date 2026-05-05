"""
12_loo.py — Leave-One-Region-Out validation for the design model only.
Methodology §3.11.3:
  - 4 main LOO iterations: one per region withheld.
  - Preprocessing (imputation + scaling) refitted per iteration on 3-region training set.
  - Hyperparameters fixed from 10_train.py (no re-tuning).
  - Ontario 7 sub-iterations: one per section (exploratory, no CIs).
Consistency check: Ontario LOO R² must be negative; if positive → leakage suspected.
Outputs: results/loo_{region}.csv
         results/loo_summary.csv
         results/ontario_section_loo.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from config import (
    RESULTS_DIR, MODELS_DIR, TARGETS, COL_REGION,
    REGIONS, PAVEMENT_FAMILIES, COL_FAMILY, SEED,
    ONTARIO_FI_GROUPS, ONTARIO_SECTION_FI,
    CONDITION_PREFIXES, GEOGRAPHIC_PROXY_COLS, REGION_PROXY_COLS,
    USE_CLIMATE_DISTANCE_LOO_WEIGHTS,
)

META_COLS = {
    "section_key", "STATE_CODE_EXP", "SHRP_ID", "PAVEMENT_FAMILY",
    "OBSERVATION_DATE", "CN_ASSIGN_DATE", "AGE_YEARS", "split",
}


def fit_impute_scale(train_df, val_df, feature_cols):
    """Refit imputation and scaling on the LOO training set only."""
    overall_med = train_df[feature_cols].median()
    family_meds = {}
    for fam in PAVEMENT_FAMILIES:
        grp = train_df[train_df[COL_FAMILY] == fam][feature_cols]
        if len(grp):
            family_meds[fam] = grp.median().fillna(overall_med)
        else:
            family_meds[fam] = overall_med

    def impute(df):
        df = df.copy()
        for fam in PAVEMENT_FAMILIES:
            mask = df[COL_FAMILY] == fam
            for col in feature_cols:
                df.loc[mask & df[col].isna(), col] = family_meds[fam].get(col, overall_med.get(col, 0))
        unknown = ~df[COL_FAMILY].isin(PAVEMENT_FAMILIES)
        for col in feature_cols:
            df.loc[unknown & df[col].isna(), col] = overall_med.get(col, 0)
        return df

    train_imp = impute(train_df)
    val_imp   = impute(val_df)

    scaler = StandardScaler()
    train_imp[feature_cols] = scaler.fit_transform(train_imp[feature_cols])
    val_imp[feature_cols]   = scaler.transform(val_imp[feature_cols])
    return train_imp, val_imp


def load_model_and_params(arch, task, target_name):
    param_f = RESULTS_DIR / f"best_params_{arch}_{task}_{target_name}.json"
    if not param_f.exists():
        return None, None
    with open(param_f) as f:
        params = json.load(f)
    return params, arch


def build_model(arch, params):
    if arch == "xgb":
        return XGBRegressor(**params, random_state=SEED, n_jobs=-1, verbosity=0)
    return RandomForestRegressor(**params, random_state=SEED, n_jobs=-1)


def loo_metrics(y_true, y_pred, region, arch, target_name):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) < 2:
        return None
    return {
        "withheld_region": region,
        "arch":            arch,
        "target":          target_name,
        "n_obs":           int(mask.sum()),
        "R2":              round(r2_score(yt, yp), 4),
        "RMSE":            round(np.sqrt(mean_squared_error(yt, yp)), 4),
        "MAE":             round(mean_absolute_error(yt, yp), 4),
    }


# Fault 5 fix: mean freeze index per region for climate-distance reweighting
REGION_MEAN_FI = {
    "Arizona": 5.9, "Georgia": 10.9, "Ohio": 316.6, "Ontario": 834.9,
}


def inverse_region_weights(train_df):
    region_counts = train_df[COL_REGION].value_counts()
    weights = train_df[COL_REGION].map(lambda r: 1.0 / region_counts[r]).values
    return weights / weights.sum() * len(weights)


def climate_distance_weights(train_df, withheld_region):
    """
    Fault 5 fix: upweight training observations whose region's freeze index
    is closest (on log scale) to the withheld region's FI.
    Returns weights that combine regional inverse-frequency with climate similarity.
    """
    fi_target = np.log1p(REGION_MEAN_FI.get(withheld_region, 1.0))
    fi_train  = train_df[COL_REGION].map(
        lambda r: np.log1p(REGION_MEAN_FI.get(r, 1.0))
    ).values
    # Gaussian similarity kernel (σ = 1.5 log-units spans ~2 orders of magnitude)
    climate_sim = np.exp(-0.5 * ((fi_train - fi_target) / 1.5) ** 2)

    # Regional inverse-frequency base weights
    region_counts = train_df[COL_REGION].value_counts()
    inv_freq = train_df[COL_REGION].map(lambda r: 1.0 / region_counts[r]).values

    combined = inv_freq * (1.0 + climate_sim)  # blend, not replace
    combined = combined / combined.sum() * len(combined)
    return combined


def run_loo_iteration(full_df, withheld_region, arch, target_name, target_col):
    """Single LOO iteration: withhold one region, train on remaining three."""
    params, _ = load_model_and_params(arch, "design", target_name)
    if params is None:
        return None

    train_df = full_df[full_df[COL_REGION] != withheld_region].copy()
    eval_df  = full_df[full_df[COL_REGION] == withheld_region].copy()

    if len(train_df) == 0 or len(eval_df) == 0:
        return None

    feature_cols = [c for c in full_df.columns
                    if c not in META_COLS
                    and c != target_col
                    and c not in GEOGRAPHIC_PROXY_COLS
                    and c not in REGION_PROXY_COLS
                    and not c.startswith(CONDITION_PREFIXES)
                    and pd.api.types.is_numeric_dtype(full_df[c])]

    train_imp, eval_imp = fit_impute_scale(train_df, eval_df, feature_cols)

    if USE_CLIMATE_DISTANCE_LOO_WEIGHTS:
        w = climate_distance_weights(train_imp, withheld_region)
    else:
        w = inverse_region_weights(train_imp)

    model = build_model(arch, params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(
            train_imp[feature_cols].fillna(0).values,
            train_imp[target_col].values,
            sample_weight=w,
        )

    y_pred = model.predict(eval_imp[feature_cols].fillna(0).values)
    y_true = eval_imp[target_col].values
    return y_true, y_pred, eval_imp


def main():
    target_name = "iri"
    target_col  = TARGETS[target_name]
    task        = "design"

    # Load the full design dataset (train+val+test combined for LOO)
    parts = []
    for sp in ["train", "val", "test"]:
        f = RESULTS_DIR / f"{task}_{target_name}_{sp}.parquet"
        if f.exists():
            parts.append(pd.read_parquet(f))
    if not parts:
        print("Design datasets not found — run 08_tasks.py first.")
        return
    full_df = pd.concat(parts, ignore_index=True)

    loo_summary = []
    all_arches  = ["xgb", "rf"]

    # ── Main 4-region LOO iterations ──────────────────────────────────────────
    for withheld in REGIONS:
        print(f"\n--- LOO: withheld = {withheld} ---")
        for arch in all_arches:
            result = run_loo_iteration(full_df, withheld, arch, target_name, target_col)
            if result is None:
                continue
            y_true, y_pred, eval_df = result
            row = loo_metrics(y_true, y_pred, withheld, arch, target_name)
            if row:
                loo_summary.append(row)
                print(f"  {arch.upper()}: R²={row['R2']:+.4f}  RMSE={row['RMSE']}")

            # Consistency check: Ontario LOO R² must be negative
            if withheld == "Ontario" and row and row["R2"] >= 0:
                print(f"  ⚠ WARNING: Ontario LOO R²={row['R2']} is NON-NEGATIVE.")
                print(f"    Possible leakage in LOO preprocessing. Investigate before reporting.")

            # Save per-region predictions
            pred_df = eval_df[["section_key", COL_REGION, "OBSERVATION_DATE",
                                target_col]].copy()
            pred_df["predicted"] = y_pred
            pred_df["residual"]  = y_true - y_pred
            pred_df["arch"]      = arch
            pred_df.to_csv(
                RESULTS_DIR / f"loo_{withheld.lower()}_{arch}.csv", index=False
            )

    # ── Ontario section-level sensitivity (7 sub-iterations) ─────────────────
    print("\n--- Ontario section-level sensitivity (7 sub-iterations) ---")
    ont_rows = []
    ontario_df = full_df[full_df[COL_REGION] == "Ontario"].copy()
    non_ontario_df = full_df[full_df[COL_REGION] != "Ontario"].copy()

    for sec_key in ontario_df["section_key"].unique():
        eval_sec  = ontario_df[ontario_df["section_key"] == sec_key].copy()
        train_sec = pd.concat(
            [non_ontario_df, ontario_df[ontario_df["section_key"] != sec_key]],
            ignore_index=True,
        )
        shrp_id = sec_key.split("_")[-1]
        fi_group = ONTARIO_FI_GROUPS.get(shrp_id, "unknown")

        for arch in all_arches:
            params, _ = load_model_and_params(arch, "design", target_name)
            if params is None:
                continue

            feature_cols = [c for c in full_df.columns
                            if c not in META_COLS
                            and c != target_col
                            and c not in GEOGRAPHIC_PROXY_COLS
                            and c not in REGION_PROXY_COLS
                            and not c.startswith(CONDITION_PREFIXES)
                            and pd.api.types.is_numeric_dtype(full_df[c])]

            train_imp, eval_imp = fit_impute_scale(train_sec, eval_sec, feature_cols)

            model = build_model(arch, params)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(
                    train_imp[feature_cols].fillna(0).values,
                    train_imp[target_col].values,
                )
            y_pred = model.predict(eval_imp[feature_cols].fillna(0).values)
            y_true = eval_imp[target_col].values

            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            r2 = r2_score(y_true[mask], y_pred[mask]) if mask.sum() >= 2 else np.nan
            # CLIM_FREEZE_INDEX was collinearity-excluded; use pre-computed raw means
            mean_fi = ONTARIO_SECTION_FI.get(str(shrp_id), np.nan)

            ont_rows.append({
                "section_key":  sec_key,
                "shrp_id":      shrp_id,
                "fi_group":     fi_group,
                "mean_fi":      round(float(mean_fi), 1) if not np.isnan(mean_fi) else None,
                "n_obs":        int(mask.sum()),
                "R2":           round(float(r2), 4) if not np.isnan(r2) else None,
                "arch":         arch,
                "note":         "exploratory — no CI, < 30 obs",
            })
            print(f"  {shrp_id} ({fi_group}) | {arch}: R²={r2:+.4f} | n={mask.sum()}")

    pd.DataFrame(ont_rows).to_csv(
        RESULTS_DIR / "ontario_section_loo.csv", index=False
    )

    # Save LOO summary
    summary_df = pd.DataFrame(loo_summary)
    summary_df.to_csv(RESULTS_DIR / "loo_summary.csv", index=False)
    print("\nLOO Summary:")
    print(summary_df[["withheld_region", "arch", "R2", "RMSE"]].to_string(index=False))
    print("\nSaved → results/loo_summary.csv")
    print("Saved → results/ontario_section_loo.csv")
    print("\n[12] LOO validation complete.\n")


if __name__ == "__main__":
    main()
