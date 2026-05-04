"""
08_tasks.py — Construct design and monitoring datasets for each target.
Methodology §3.8:
  Design task   → structural + traffic + climate inputs only (no prior condition).
  Monitoring task → design inputs + date-aware temporal lag of prior target value.
Lag windows: IRI 730d, Rutting 730d, Distress 1095d.
Sections with no prior observation within the window are excluded from monitoring.
Monitoring viability check: ≥150 obs across ≥20 sections required to proceed.
Outputs: results/{task}_{target}_{split}.parquet  (task: design / monitoring)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from config import (
    RESULTS_DIR, TARGETS, COL_DATE, COL_SECTION, COL_REGION,
    LAG_WINDOWS, MONITOR_MIN_OBS, MONITOR_MIN_SECTIONS,
)

META_COLS = {
    "section_key", "STATE_CODE_EXP", "SHRP_ID", "PAVEMENT_FAMILY",
    "OBSERVATION_DATE", "CN_ASSIGN_DATE", "AGE_YEARS", "split",
}

# Fault 1/3 fix: ordinal climate-zone encoding gives model explicit regime context.
# Ordering matches freeze-severity gradient so encoding is ordinally meaningful.
CLIMATE_ZONE_MAP = {
    "Arizona": 1,  # arid/hot            (FI ≈ 5.9)
    "Georgia": 2,  # warm-humid subtrop. (FI ≈ 10.9)
    "Ohio":    3,  # moderate freeze     (FI ≈ 316.6)
    "Ontario": 4,  # extreme freeze      (FI ≈ 834.9)
}


def build_lag_feature(
    df: pd.DataFrame,
    target_col: str,
    window_days: int,
) -> pd.DataFrame:
    """
    For each observation, look up the most recent prior value of target_col
    within the preceding window_days on the same section.
    Observations with no valid prior are returned with lag = NaN.
    """
    df = df.sort_values(["section_key", COL_DATE]).copy()
    lag_col = f"LAG_{target_col}"
    df[lag_col] = np.nan

    for sec_key, grp in df.groupby("section_key", sort=False):
        grp = grp.sort_values(COL_DATE)
        dates  = grp[COL_DATE].values
        values = grp[target_col].values
        indices = grp.index.tolist()

        for i in range(1, len(grp)):
            cur_date = dates[i]
            window   = pd.Timedelta(days=window_days)
            prior_mask = (dates[:i] < cur_date) & (dates[:i] >= cur_date - window)
            if prior_mask.any():
                last_idx = np.where(prior_mask)[0][-1]
                df.at[indices[i], lag_col] = values[last_idx]

    return df


def check_monitoring_viability(
    df: pd.DataFrame,
    lag_col: str,
    target_name: str,
) -> bool:
    valid = df[df[lag_col].notna()]
    n_obs  = len(valid)
    n_secs = valid["section_key"].nunique()
    ok = (n_obs >= MONITOR_MIN_OBS) and (n_secs >= MONITOR_MIN_SECTIONS)
    status = "OK" if ok else "INFEASIBLE — design model only"
    print(f"  Monitoring viability: {n_obs} obs, {n_secs} sections → {status}")
    return ok


def main():
    for target_name, target_col in TARGETS.items():
        # Load all three splits
        splits = {}
        missing = False
        for sp in ["train", "val", "test"]:
            src = RESULTS_DIR / f"selected_{target_name}_{sp}.parquet"
            if not src.exists():
                print(f"[{target_name}] {src.name} not found — run 07_collinearity.py first")
                missing = True
                break
            splits[sp] = pd.read_parquet(src)

        if missing:
            continue

        window = LAG_WINDOWS[target_name]
        print(f"\n[{target_name}] lag window = {window} days")

        # Inject CLIMATE_ZONE_CODE into every split (Fault 1/3 fix)
        for sp in splits:
            splits[sp] = splits[sp].copy()
            splits[sp]["CLIMATE_ZONE_CODE"] = (
                splits[sp][COL_REGION].map(CLIMATE_ZONE_MAP).fillna(0).astype(float)
            )

        # ── Design task ───────────────────────────────────────────────────────
        for sp, df in splits.items():
            out = RESULTS_DIR / f"design_{target_name}_{sp}.parquet"
            df.to_parquet(out, index=False)
        print(f"  Design datasets saved (CLIMATE_ZONE_CODE injected).")

        # ── Monitoring task: build lag on combined data, then split ────────────
        combined = pd.concat(splits.values(), ignore_index=True)
        combined = build_lag_feature(combined, target_col, window)
        lag_col  = f"LAG_{target_col}"

        # Fault 2 fix: add ΔIRI = current − lag (residual deterioration increment)
        delta_col = f"DELTA_{target_col}"
        combined[delta_col] = combined[target_col] - combined[lag_col]

        viable = check_monitoring_viability(
            combined[combined["split"] == "train"], lag_col, target_name
        )

        if not viable:
            print(f"  [{target_name}] Monitoring task SKIPPED — insufficient sample.")
            continue

        for sp in ["train", "val", "test"]:
            sub = combined[combined["split"] == sp].copy()
            # Exclude observations with no valid lag
            sub_monitor = sub[sub[lag_col].notna()].copy()
            out = RESULTS_DIR / f"monitoring_{target_name}_{sp}.parquet"
            sub_monitor.to_parquet(out, index=False)

            n_excl = len(sub) - len(sub_monitor)
            print(f"  monitoring {sp:5s}: {len(sub_monitor)} obs "
                  f"({n_excl} excluded — no prior within {window}d)")

            # Also save a Δ-target version for skill-score monitoring (Fault 2)
            delta_monitor = sub_monitor[sub_monitor[delta_col].notna()].copy()
            delta_out = RESULTS_DIR / f"monitoring_delta_{target_name}_{sp}.parquet"
            delta_monitor.to_parquet(delta_out, index=False)

    print("\n[08] Task construction complete.\n")


if __name__ == "__main__":
    main()
