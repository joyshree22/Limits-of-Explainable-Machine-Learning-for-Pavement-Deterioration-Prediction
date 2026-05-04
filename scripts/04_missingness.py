"""
04_missingness.py — Apply 50% missingness threshold independently per target dataset.
Methodology §3.4: threshold applied at feature level, not observation level.
Thresholds: IRI > 325, Rutting > 283, Distress > 189 missing observations → drop feature.
Outputs: results/dataset_{target}_thresholded.parquet
         results/features_retained_{target}.txt
         results/features_excluded_{target}.txt
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from config import RESULTS_DIR, TARGETS, MISS_THRESHOLD

META_COLS = {
    "section_key", "STATE_CODE_EXP", "SHRP_ID", "PAVEMENT_FAMILY",
    "OBSERVATION_DATE", "CN_ASSIGN_DATE", "AGE_YEARS",
}


def apply_threshold(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, list, list]:
    n_obs = len(df)
    threshold = int(n_obs * MISS_THRESHOLD)

    feature_cols = [c for c in df.columns
                    if c not in META_COLS and c != target_col]

    retained, excluded = [], []
    for col in feature_cols:
        n_missing = df[col].isna().sum()
        if n_missing > threshold:
            excluded.append(col)
        else:
            retained.append(col)

    keep_cols = list(META_COLS & set(df.columns)) + [target_col, "AGE_YEARS"] + retained
    # preserve actual column order
    keep_cols = [c for c in df.columns if c in set(keep_cols)]
    return df[keep_cols].copy(), retained, excluded


def main():
    for target_name, target_col in TARGETS.items():
        src = RESULTS_DIR / f"dataset_{target_name}_features.parquet"
        if not src.exists():
            print(f"[{target_name}] source not found — run 03_features.py first")
            continue

        df = pd.read_parquet(src)
        n_obs = len(df)
        threshold = int(n_obs * MISS_THRESHOLD)

        df_thresh, retained, excluded = apply_threshold(df, target_col)

        out = RESULTS_DIR / f"dataset_{target_name}_thresholded.parquet"
        df_thresh.to_parquet(out, index=False)

        ret_path = RESULTS_DIR / f"features_retained_{target_name}.txt"
        exc_path = RESULTS_DIR / f"features_excluded_{target_name}.txt"
        ret_path.write_text("\n".join(retained))
        exc_path.write_text("\n".join(excluded))

        print(f"[{target_name}] n_obs={n_obs}, threshold=>{threshold} missing")
        print(f"  Retained : {len(retained)} features")
        print(f"  Excluded : {len(excluded)} features")
        print(f"  Saved    → {out.name}")

    print("\n[04] Missingness threshold complete.\n")


if __name__ == "__main__":
    main()
