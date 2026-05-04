"""
05_partition.py — Section-wise stratified train / validation / test split.
Methodology §3.5: every observation from a section goes to exactly one partition.
Stratified by region to ensure all four climate zones appear in each partition.
Split: 34 train / 7 val / 7 test sections (by region: AZ 16/3/3, GA 8/2/2, OH 5/1/1, ON 5/1/1).
Outputs: results/splits_{target}.parquet  (adds 'split' column: train/val/test)
         results/section_assignment.csv   (section → split, shared across targets)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from config import (
    RESULTS_DIR, TARGETS, COL_REGION, COL_SECTION,
    REGION_SPLIT, SEED,
)

META_COLS = {
    "section_key", "STATE_CODE_EXP", "SHRP_ID", "PAVEMENT_FAMILY",
    "OBSERVATION_DATE", "CN_ASSIGN_DATE", "AGE_YEARS",
}


def assign_sections(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Returns a DataFrame mapping (region, section_key) → split label.
    Stratified by region; uses a fixed RNG for reproducibility.
    """
    rows = []
    for region, (n_train, n_val, n_test) in REGION_SPLIT.items():
        sections = (
            df[df[COL_REGION] == region]["section_key"]
            .unique()
            .tolist()
        )
        sections_arr = np.array(sorted(sections))
        rng.shuffle(sections_arr)

        if len(sections_arr) != n_train + n_val + n_test:
            raise ValueError(
                f"Region {region}: expected {n_train+n_val+n_test} sections, "
                f"found {len(sections_arr)}"
            )

        for sec in sections_arr[:n_train]:
            rows.append({"region": region, "section_key": sec, "split": "train"})
        for sec in sections_arr[n_train:n_train + n_val]:
            rows.append({"region": region, "section_key": sec, "split": "val"})
        for sec in sections_arr[n_train + n_val:]:
            rows.append({"region": region, "section_key": sec, "split": "test"})

    return pd.DataFrame(rows)


def main():
    rng = np.random.default_rng(SEED)

    # Build section assignment from first target (same sections across all targets)
    first_target = "iri"
    src = RESULTS_DIR / f"dataset_{first_target}_thresholded.parquet"
    if not src.exists():
        print("Run 04_missingness.py first.")
        return

    df_ref = pd.read_parquet(src, columns=["section_key", COL_REGION])
    assignment = assign_sections(df_ref, rng)
    assignment.to_csv(RESULTS_DIR / "section_assignment.csv", index=False)

    split_counts = assignment["split"].value_counts()
    print("Section assignment:")
    for split_name in ["train", "val", "test"]:
        print(f"  {split_name}: {split_counts.get(split_name, 0)} sections")
    print()

    # Apply assignment to each target dataset
    for target_name, target_col in TARGETS.items():
        src = RESULTS_DIR / f"dataset_{target_name}_thresholded.parquet"
        if not src.exists():
            print(f"[{target_name}] source not found — skipping")
            continue

        df = pd.read_parquet(src)
        df = df.merge(assignment[["section_key", "split"]], on="section_key", how="left")

        unmapped = df["split"].isna().sum()
        if unmapped > 0:
            print(f"  WARNING: {unmapped} rows have no split assignment for {target_name}")

        out = RESULTS_DIR / f"splits_{target_name}.parquet"
        df.to_parquet(out, index=False)

        for sp in ["train", "val", "test"]:
            n = (df["split"] == sp).sum()
            n_sec = df[df["split"] == sp]["section_key"].nunique()
            print(f"[{target_name}] {sp:5s}: {n:4d} obs, {n_sec} sections")
        print()

    print("[05] Partitioning complete.\n")


if __name__ == "__main__":
    main()
