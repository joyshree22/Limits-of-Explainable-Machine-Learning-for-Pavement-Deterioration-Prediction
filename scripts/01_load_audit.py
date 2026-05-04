"""
01_load_audit.py — Load raw data, parse dates, produce missingness audit.
Methodology §3.1: record exact row/column counts before any transformation.
Outputs: results/audit_raw.csv, results/missingness_audit.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from config import (
    RAW_CSV, RESULTS_DIR, TARGETS, COL_REGION, COL_SECTION,
    COL_DATE, COL_CN_DATE, COL_FAMILY, REGIONS,
)


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV, low_memory=False)
    df[COL_DATE]    = pd.to_datetime(df[COL_DATE],    errors="coerce")
    df[COL_CN_DATE] = pd.to_datetime(df[COL_CN_DATE], errors="coerce")
    return df


def audit(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("RAW DATA AUDIT")
    print("=" * 60)
    print(f"Total rows    : {len(df):,}")
    print(f"Total columns : {df.shape[1]:,}")
    print(f"Date range    : {df[COL_DATE].min().date()} → {df[COL_DATE].max().date()}")

    print("\n--- Sections per region ---")
    for region in REGIONS:
        n = df[df[COL_REGION] == region][COL_SECTION].nunique()
        print(f"  {region:<12}: {n} sections")

    print("\n--- Rows per region ---")
    for region in REGIONS:
        n = (df[COL_REGION] == region).sum()
        print(f"  {region:<12}: {n:,} rows")

    print("\n--- Target variable availability ---")
    for name, col in TARGETS.items():
        n = df[col].notna().sum()
        print(f"  {name:<10}: {n:,} non-null  ({n/len(df)*100:.1f}%)")

    print("\n--- Pavement family ---")
    for fam, cnt in df[COL_FAMILY].value_counts().items():
        print(f"  {fam:<8}: {cnt:,} rows")

    fi = df.get("CLIM_FREEZE_INDEX", pd.Series(dtype=float))
    fi_zero = (fi == 0).sum()
    print(f"\n--- Freeze Index (CLIM_FREEZE_INDEX) ---")
    print(f"  FI = 0 records: {fi_zero:,} ({fi_zero/len(df)*100:.1f}%)")
    for region in REGIONS:
        vals = df.loc[df[COL_REGION] == region, "CLIM_FREEZE_INDEX"].dropna()
        if len(vals):
            print(f"  {region:<12}: mean={vals.mean():.1f}, "
                  f"min={vals.min():.1f}, max={vals.max():.1f}")

    # Missingness audit — all columns
    miss_pct = df.isna().mean().mul(100).round(2)
    miss_df = miss_pct.reset_index()
    miss_df.columns = ["feature", "pct_missing"]
    miss_df = miss_df.sort_values("pct_missing", ascending=False)
    miss_df.to_csv(RESULTS_DIR / "missingness_audit.csv", index=False)
    print(f"\n  Features > 50% missing : {(miss_pct > 50).sum()}")
    print(f"  Overall missing cells  : {df.isna().mean().mean()*100:.1f}%")

    # Row-level audit summary
    summary = {
        "total_rows":       len(df),
        "total_cols":       df.shape[1],
        "date_min":         str(df[COL_DATE].min().date()),
        "date_max":         str(df[COL_DATE].max().date()),
        "iri_nonnull":      int(df[TARGETS["iri"]].notna().sum()),
        "rutting_nonnull":  int(df[TARGETS["rutting"]].notna().sum()),
        "distress_nonnull": int(df[TARGETS["distress"]].notna().sum()),
        "fi_zero_count":    int(fi_zero),
    }
    pd.Series(summary).to_csv(RESULTS_DIR / "audit_raw.csv", header=["value"])
    print("\nSaved → results/audit_raw.csv")
    print("Saved → results/missingness_audit.csv")


def main():
    df = load_raw()
    audit(df)
    print("\n[01] Complete.\n")


if __name__ == "__main__":
    main()
