"""
06_impute_scale.py — Group-median imputation (by pavement family) + StandardScaler.
Methodology §3.6: all parameters fitted on training sections ONLY, then applied to val/test.
Outputs: results/processed_{target}_{split}.parquet  (train / val / test)
         models/imputer_{target}.joblib
         models/scaler_{target}.joblib
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from config import (
    RESULTS_DIR, MODELS_DIR, TARGETS, COL_FAMILY,
    PAVEMENT_FAMILIES,
)

META_COLS = {
    "section_key", "STATE_CODE_EXP", "SHRP_ID", "PAVEMENT_FAMILY",
    "OBSERVATION_DATE", "CN_ASSIGN_DATE", "AGE_YEARS", "split",
}


def fit_imputer(train_df: pd.DataFrame, feature_cols: list) -> dict:
    """
    Compute group-median per (family, feature) on training rows only.
    Falls back to overall training median if a family group is absent.
    Returns a dict: {family: {feature: median_value}}
    """
    medians = {}
    overall = train_df[feature_cols].median()

    for family in PAVEMENT_FAMILIES:
        group = train_df[train_df[COL_FAMILY] == family][feature_cols]
        if len(group) == 0:
            medians[family] = overall.to_dict()
        else:
            group_med = group.median()
            # fill any family-level NaN with overall median
            group_med = group_med.fillna(overall)
            medians[family] = group_med.to_dict()

    medians["__overall__"] = overall.to_dict()
    return medians


def apply_imputer(df: pd.DataFrame, medians: dict, feature_cols: list) -> pd.DataFrame:
    df = df.copy()
    for family in PAVEMENT_FAMILIES:
        mask = df[COL_FAMILY] == family
        if mask.any():
            fill_vals = medians.get(family, medians["__overall__"])
            for col in feature_cols:
                if col in fill_vals:
                    df.loc[mask & df[col].isna(), col] = fill_vals[col]
    # rows with unknown family → overall median
    unknown_mask = ~df[COL_FAMILY].isin(PAVEMENT_FAMILIES)
    if unknown_mask.any():
        for col in feature_cols:
            df.loc[unknown_mask & df[col].isna(), col] = medians["__overall__"].get(col, np.nan)
    return df


def main():
    for target_name, target_col in TARGETS.items():
        src = RESULTS_DIR / f"splits_{target_name}.parquet"
        if not src.exists():
            print(f"[{target_name}] splits not found — run 05_partition.py first")
            continue

        df = pd.read_parquet(src)
        feature_cols = [c for c in df.columns
                        if c not in META_COLS
                        and c != target_col
                        and pd.api.types.is_numeric_dtype(df[c])]

        train_df = df[df["split"] == "train"].copy()
        val_df   = df[df["split"] == "val"].copy()
        test_df  = df[df["split"] == "test"].copy()

        # ── Imputation ────────────────────────────────────────────────────────
        medians = fit_imputer(train_df, feature_cols)
        joblib.dump(medians, MODELS_DIR / f"imputer_{target_name}.joblib")

        train_df = apply_imputer(train_df, medians, feature_cols)
        val_df   = apply_imputer(val_df,   medians, feature_cols)
        test_df  = apply_imputer(test_df,  medians, feature_cols)

        # ── Scaling (StandardScaler, fit on train only) ───────────────────────
        scaler = StandardScaler()
        train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
        val_df[feature_cols]   = scaler.transform(val_df[feature_cols])
        test_df[feature_cols]  = scaler.transform(test_df[feature_cols])
        joblib.dump(scaler, MODELS_DIR / f"scaler_{target_name}.joblib")

        # ── Save ──────────────────────────────────────────────────────────────
        for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            out = RESULTS_DIR / f"processed_{target_name}_{split_name}.parquet"
            split_df.to_parquet(out, index=False)

        remaining_nan = {
            sp: df_sp[feature_cols].isna().sum().sum()
            for sp, df_sp in [("train", train_df), ("val", val_df), ("test", test_df)]
        }
        print(f"[{target_name}] imputed + scaled | features={len(feature_cols)}")
        print(f"  Residual NaN after imputation: {remaining_nan}")

    print("\n[06] Imputation and scaling complete.\n")


if __name__ == "__main__":
    main()
