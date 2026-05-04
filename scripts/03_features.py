"""
03_features.py — Feature engineering: domain assembly + 9 physics-based composites.
Methodology §3.3: construct the candidate feature set before any split or threshold.
All composite formulas are fixed here; no split statistics are used.
Outputs: results/dataset_{target}_features.parquet  (one per target)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from config import RESULTS_DIR, TARGETS


META_COLS = {
    "section_key", "STATE_CODE_EXP", "SHRP_ID", "PAVEMENT_FAMILY",
    "OBSERVATION_DATE", "CN_ASSIGN_DATE",
}


def build_composites(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nine physics-based composite features (Methodology §3.3).
    CLIM_FREEZE_INDEX = 0 for 32.3% of records (AZ + GA).
    The structural-freeze-insulation composite uses 1/(1+FI) — NOT ratio form —
    to avoid division-by-zero for those 1,419 observations.
    """
    fi  = df.get("CLIM_FREEZE_INDEX",    pd.Series(0.0, index=df.index))
    ft  = df.get("CLIM_FREEZE_THAW",     pd.Series(0.0, index=df.index))
    pr  = df.get("CLIM_PRECIPITATION",   pd.Series(0.0, index=df.index))
    ac  = df.get("LAYER_THICKNESS_AC_MM",pd.Series(np.nan, index=df.index))
    bsg = df.get("AC_BSG",               pd.Series(np.nan, index=df.index))
    age = df.get("AGE_YEARS",            pd.Series(np.nan, index=df.index))
    nly = df.get("LAYER_COUNT_AC",       pd.Series(np.nan, index=df.index))
    aadt= df.get("TRF_TVP_AADTT_FIRST_YEAR_LTPP_LANE", pd.Series(np.nan, index=df.index))
    mr  = df.get("UB_RESILIENT_MODULUS", pd.Series(np.nan, index=df.index))
    ts  = df.get("CLIM_TEMP_MEAN_AVG_SUMMER", pd.Series(np.nan, index=df.index))
    tw  = df.get("CLIM_TEMP_MEAN_AVG_WINTER", pd.Series(np.nan, index=df.index))

    composites = {
        "COMP_WET_FREEZE":           ft * pr,
        "COMP_STRUCT_FREEZE_INS":    ac * (1.0 / (1.0 + fi)),   # corrected: no div-by-zero
        "COMP_MAT_FREEZE_SUSC":      ft * bsg,
        "COMP_CUMUL_WET_FREEZE":     fi * pr * age,
        "COMP_FREEZE_PER_LAYER":     fi / nly.replace(0, np.nan),
        "COMP_TRAFFIC_CLIMATE":      aadt * fi,
        "COMP_STRUCT_ADEQUACY":      ac / mr.replace(0, np.nan),
        "COMP_THERMAL_GRADIENT":     (ts - tw) * fi,
        "COMP_AGE_CLIMATE":          age * fi,
    }
    for name, series in composites.items():
        df[name] = series
    return df


def main():
    for target_name in TARGETS:
        src = RESULTS_DIR / f"dataset_{target_name}_visit.parquet"
        if not src.exists():
            print(f"[{target_name}] source not found — run 02_aggregate.py first")
            continue

        df = pd.read_parquet(src)
        n_before = df.shape[1]
        df = build_composites(df)
        n_after = df.shape[1]

        out = RESULTS_DIR / f"dataset_{target_name}_features.parquet"
        df.to_parquet(out, index=False)

        feature_cols = [c for c in df.columns if c not in META_COLS
                        and c != TARGETS[target_name] and c != "AGE_YEARS"]
        comp_cols    = [c for c in df.columns if c.startswith("COMP_")]
        print(f"[{target_name}] +{n_after - n_before} composite cols | "
              f"total columns={df.shape[1]} | composites={len(comp_cols)} | {out.name}")

    print("\n[03] Feature engineering complete.\n")


if __name__ == "__main__":
    main()
