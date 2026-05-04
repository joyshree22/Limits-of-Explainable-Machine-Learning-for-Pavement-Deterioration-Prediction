"""
02_aggregate.py — Visit-level aggregation for IRI; pass-through for rutting/distress.
Methodology §3.2: ~5 measurement runs per site visit → aggregate to visit mean.
IRI: 3,233 raw rows → 650 unique visit-dates.
Rutting/Distress: single measurement per visit; no aggregation needed.
Outputs: results/dataset_{target}_visit.parquet  (one file per target)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from config import (
    RAW_CSV, RESULTS_DIR, TARGETS, COL_REGION, COL_SECTION,
    COL_DATE, COL_FAMILY, COL_CN_DATE,
)

NON_FEATURE_COLS = {
    # identifiers and dates — kept as metadata, never used as features
    "STATE_CODE", "STATE_CODE_EXP", "SHRP_ID", "CONSTRUCTION_NO",
    "VISIT_NO", "RUN_NUMBER", "OBSERVATION_DATE", "OBSERVATION_YEAR",
    "SOURCE", "CN_ASSIGN_DATE", "ASSIGN_DATE", "DEASSIGN_DATE",
    "PAVEMENT_FAMILY_EXP", "CN_CHANGE_REASON", "GPS_SPS",
    "EXPERIMENT_NO", "SEAS_ID", "EXP_SECT_RS",
    # all target columns (excluded from feature set regardless of which target is modeled)
    "IRI_MRI", "IRI_IRI_LEFT_WHEEL_PATH", "IRI_IRI_RIGHT_WHEEL_PATH", "IRI_IRI_CENTER_LANE",
    "RUT_LLH_DEPTH_1_8_MEAN", "RUT_RLH_DEPTH_1_8_MEAN",
    "RUT_MAX_MEAN_DEPTH_1_8", "RUT_LLH_DEPTH_WIRE_REF_MEAN",
    "RUT_RLH_DEPTH_WIRE_REF_MEAN", "RUT_MAX_MEAN_DEPTH_WIRE_REF",
    "RUT_T_PROF_DEVICE_CODE", "RUT_NO_PROFILES", "RUT_START_TIME", "RUT_PVMT_WIDTH",
    "DIS_HPMS16_CRACKING_PERCENT_AC", "DIS_MEPDG_CRACKING_PERCENT_AC",
    "DIS_MEPDG_TRANS_CRACK_LENGTH_AC", "DIS_MEPDG_LONG_CRACK_LENGTH_AC",
    "DIS_ME_PERCENT_WHEEL_PATH_CRACK",
}


def load_and_prepare() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV, low_memory=False)
    df[COL_DATE]    = pd.to_datetime(df[COL_DATE],    errors="coerce")
    df[COL_CN_DATE] = pd.to_datetime(df[COL_CN_DATE], errors="coerce")
    return df


def make_section_key(df: pd.DataFrame) -> pd.Series:
    """Composite section key: region + SHRP_ID (SHRP_IDs are not globally unique)."""
    return df[COL_REGION].astype(str) + "_" + df[COL_SECTION].astype(str)


def derive_age(df: pd.DataFrame) -> pd.Series:
    return (df[COL_DATE] - df[COL_CN_DATE]).dt.days / 365.25


def aggregate_target(df: pd.DataFrame, target_name: str) -> pd.DataFrame:
    target_col = TARGETS[target_name]
    sub = df[df[target_col].notna()].copy()
    sub["section_key"] = make_section_key(sub)
    sub["AGE_YEARS"]   = derive_age(sub)

    # Group-by (section, visit date) → take column means for numeric cols
    # Non-numeric and identifier cols: take first value within group
    id_cols  = ["section_key", COL_REGION, COL_SECTION, COL_FAMILY, COL_DATE, COL_CN_DATE]
    num_cols = [c for c in sub.columns
                if c not in NON_FEATURE_COLS
                and c not in id_cols
                and c != "AGE_YEARS"
                and pd.api.types.is_numeric_dtype(sub[c])]

    agg_dict = {c: "mean" for c in num_cols}
    agg_dict[target_col] = "mean"
    agg_dict["AGE_YEARS"] = "mean"

    # Identifier columns: first value
    for c in [COL_REGION, COL_SECTION, COL_FAMILY, COL_CN_DATE]:
        agg_dict[c] = "first"

    grouped = (
        sub.groupby(["section_key", COL_DATE], sort=True)
        .agg(agg_dict)
        .reset_index()
    )
    grouped = grouped.sort_values(["section_key", COL_DATE]).reset_index(drop=True)
    return grouped


def main():
    df = load_and_prepare()

    for target_name in TARGETS:
        agg = aggregate_target(df, target_name)
        out = RESULTS_DIR / f"dataset_{target_name}_visit.parquet"
        agg.to_parquet(out, index=False)
        n_raw = df[TARGETS[target_name]].notna().sum()
        print(f"[{target_name}] raw={n_raw:,} rows → {len(agg):,} visit-level observations | {out.name}")

    print("\n[02] Aggregation complete.\n")


if __name__ == "__main__":
    main()
