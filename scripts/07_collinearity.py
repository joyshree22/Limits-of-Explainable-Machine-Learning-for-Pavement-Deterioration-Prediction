"""
07_collinearity.py — Spearman hierarchical clustering to reduce collinearity.
Methodology §3.7: distance = 1 − |ρ|, average linkage, cut at distance 0.15 (|ρ| = 0.85).
Within each cluster, retain one representative using the priority hierarchy + tiebreaker.
Outputs: results/selected_{target}_{split}.parquet
         results/features_selected_{target}.txt
         results/collinearity_clusters_{target}.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from config import (
    RESULTS_DIR, TARGETS, SPEARMAN_CORR_THRESHOLD,
    CONDITION_PREFIXES, GEOGRAPHIC_PROXY_COLS, REGION_PROXY_COLS,
)

META_COLS = {
    "section_key", "STATE_CODE_EXP", "SHRP_ID", "PAVEMENT_FAMILY",
    "OBSERVATION_DATE", "CN_ASSIGN_DATE", "AGE_YEARS", "split",
}

# Fault 6 fix: force-retain regardless of clustering outcome (physics-mandatory)
PHYSICS_FORCE_RETAIN = {"CLIM_FREEZE_INDEX"}

# Exclude proxies and any same-visit condition measurements that survive earlier
# stages. The latter should already be removed in 02_aggregate.py, but keeping the
# guard here makes leakage fail closed.
PRIMARY_EXCLUDE = GEOGRAPHIC_PROXY_COLS | REGION_PROXY_COLS

# Priority tier — lower number = higher priority for retention
def feature_priority(col: str) -> int:
    if col.startswith("COMP_"):
        return 0   # physics-based composite
    if col.startswith("CLIM_") or col.startswith("LAYER_") or col.startswith("AC_BSG"):
        return 1   # directly measured structural or climate
    if col.startswith("TRF_ALDF"):
        return 3   # raw load distribution bin (expect large cluster)
    return 2       # derived aggregate or raw measurement


def select_cluster_representative(
    cluster_features: list[str],
    feature_cols: list[str],
    missingness: pd.Series,
    target_correlations: pd.Series,
) -> str:
    """
    Priority: composite > direct structural/climate > derived > raw
    Tiebreaker: highest |Spearman ρ| with the target on training set.
    """
    candidates = pd.DataFrame({
        "feature":     cluster_features,
        "priority":    [feature_priority(f) for f in cluster_features],
        "missing_pct": [missingness.get(f, 0.0) for f in cluster_features],
        "target_corr": [abs(target_correlations.get(f, 0.0)) for f in cluster_features],
    })
    candidates = candidates.sort_values(
        ["priority", "missing_pct", "target_corr"],
        ascending=[True, True, False]
    )
    return candidates.iloc[0]["feature"]


def reduce_collinearity(
    train_df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
) -> tuple[list[str], pd.DataFrame]:
    """
    Returns (selected_features, cluster_map_df).
    Spearman correlation matrix computed on training set only.
    """
    X = train_df[feature_cols].copy()

    # Spearman correlation — handle near-constant columns gracefully
    corr_matrix, _ = spearmanr(X.values, nan_policy="omit")
    if corr_matrix.ndim == 0:
        corr_matrix = np.array([[1.0]])
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    np.fill_diagonal(corr_matrix, 1.0)

    # Convert to distance matrix
    dist_matrix = 1.0 - np.abs(corr_matrix)
    dist_matrix = np.clip(dist_matrix, 0.0, 1.0)
    np.fill_diagonal(dist_matrix, 0.0)

    # Hierarchical clustering
    condensed = squareform(dist_matrix, checks=False)
    Z = linkage(condensed, method="average")
    distance_threshold = 1.0 - SPEARMAN_CORR_THRESHOLD   # 0.15
    labels = fcluster(Z, t=distance_threshold, criterion="distance")

    # Missingness and target correlation for tiebreaker
    missingness     = X.isna().mean()
    target_corr, _  = spearmanr(
        X.values, train_df[target_col].values, axis=0, nan_policy="omit"
    )
    if isinstance(target_corr, np.ndarray):
        # last column is correlation with target
        target_corr_series = pd.Series(target_corr[:-1, -1], index=feature_cols)
    else:
        target_corr_series = pd.Series([target_corr], index=feature_cols)

    # Build cluster map and select representatives
    cluster_df_rows = []
    selected = []
    for cluster_id in np.unique(labels):
        members = [feature_cols[i] for i, l in enumerate(labels) if l == cluster_id]
        rep = select_cluster_representative(
            members, feature_cols, missingness, target_corr_series
        )
        selected.append(rep)
        for m in members:
            cluster_df_rows.append({
                "cluster_id":   cluster_id,
                "feature":      m,
                "representative": rep,
                "is_selected":  m == rep,
            })

    cluster_df = pd.DataFrame(cluster_df_rows)
    return sorted(selected), cluster_df


def main():
    for target_name, target_col in TARGETS.items():
        train_src = RESULTS_DIR / f"processed_{target_name}_train.parquet"
        if not train_src.exists():
            print(f"[{target_name}] processed train not found — run 06_impute_scale.py first")
            continue

        train_df = pd.read_parquet(train_src)
        feature_cols = [c for c in train_df.columns
                        if c not in META_COLS
                        and c != target_col
                        and c not in PRIMARY_EXCLUDE
                        and not c.startswith(CONDITION_PREFIXES)
                        and pd.api.types.is_numeric_dtype(train_df[c])
                        and train_df[c].notna().sum() > 0]

        selected, cluster_df = reduce_collinearity(train_df, target_col, feature_cols)

        # Force-add physics-mandatory features excluded by clustering tiebreaker
        for feat in PHYSICS_FORCE_RETAIN:
            if feat in train_df.columns and feat not in selected:
                selected.append(feat)
                cluster_df = pd.concat([
                    cluster_df,
                    pd.DataFrame([{
                        "cluster_id": "force_retained",
                        "feature": feat,
                        "representative": feat,
                        "is_selected": True,
                    }]),
                ], ignore_index=True)
                print(f"  Force-retained: {feat} (physics-mandatory)")

        leaked = [f for f in selected if f.startswith(CONDITION_PREFIXES)]
        if leaked:
            raise ValueError(
                f"[{target_name}] condition-measure leakage in selected features: {leaked[:10]}"
            )

        # Save cluster map
        cluster_df.to_csv(
            RESULTS_DIR / f"collinearity_clusters_{target_name}.csv", index=False
        )
        (RESULTS_DIR / f"features_selected_{target_name}.txt").write_text(
            "\n".join(sorted(selected))
        )

        n_clusters = cluster_df["cluster_id"].nunique()
        proxy_excl = [c for c in PRIMARY_EXCLUDE
                    if c in train_df.columns]
        print(f"[{target_name}] {len(feature_cols)} features (excl. {proxy_excl}) "
              f"→ {len(selected)} selected ({n_clusters} clusters)")

        # Apply selection to train / val / test
        keep_cols = (
            [c for c in train_df.columns if c in META_COLS]
            + [target_col]
            + [c for c in selected if c in train_df.columns]
        )
        keep_cols = list(dict.fromkeys(keep_cols))   # preserve order, deduplicate

        for split_name in ["train", "val", "test"]:
            src = RESULTS_DIR / f"processed_{target_name}_{split_name}.parquet"
            out = RESULTS_DIR / f"selected_{target_name}_{split_name}.parquet"
            if src.exists():
                split_df = pd.read_parquet(src)
                valid_keep = [c for c in keep_cols if c in split_df.columns]
                split_df[valid_keep].to_parquet(out, index=False)

        print(f"  Saved selected_{target_name}_{{train,val,test}}.parquet")

    print("\n[07] Collinearity reduction complete.\n")


if __name__ == "__main__":
    main()
