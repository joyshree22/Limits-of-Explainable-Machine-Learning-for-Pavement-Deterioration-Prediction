"""
09_benchmarks.py — Four benchmark models + persistence at k=1,2,3 years.
Methodology §3.9: all benchmarks evaluated on a COMMON evaluation set
(observations with a prior within 365d — the most restrictive persistence window).
Outputs: results/benchmark_metrics.csv
         results/persistence_metrics.csv
         results/common_set_comparison.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import warnings
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from config import (
    RESULTS_DIR, TARGETS, COL_DATE, PERSISTENCE_YEARS,
    BOOTSTRAP_N, BOOTSTRAP_CI, SEED,
)

META_COLS = {
    "section_key", "STATE_CODE_EXP", "SHRP_ID", "PAVEMENT_FAMILY",
    "OBSERVATION_DATE", "CN_ASSIGN_DATE", "AGE_YEARS", "split",
}


def metrics(y_true, y_pred, label: str) -> dict:
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[mask], y_pred[mask]
    return {
        "model": label,
        "n":     int(mask.sum()),
        "R2":    round(r2_score(yt, yp), 4),
        "RMSE":  round(np.sqrt(mean_squared_error(yt, yp)), 4),
        "MAE":   round(mean_absolute_error(yt, yp), 4),
    }


def aashto_func(age, a, b):
    return a * np.exp(b * age)


def build_persistence_lookup(
    train_df: pd.DataFrame,
    target_col: str,
    window_days: int,
) -> dict:
    """Pre-build {(section_key, obs_date) → prior_value} for speed."""
    lookup = {}
    train_df = train_df.sort_values(["section_key", COL_DATE])
    for sec_key, grp in train_df.groupby("section_key", sort=False):
        grp = grp.sort_values(COL_DATE)
        dates  = grp[COL_DATE].values
        values = grp[target_col].values
        for i, (d, v) in enumerate(zip(dates, values)):
            window = pd.Timedelta(days=window_days)
            prior_mask = (dates[:i] < d) & (dates[:i] >= d - window)
            if prior_mask.any():
                last = values[np.where(prior_mask)[0][-1]]
                lookup[(sec_key, pd.Timestamp(d))] = last
    return lookup


def persistence_predict(test_df, target_col, window_days):
    # Priors come from earlier visits within the same test section (section-wise split)
    lookup = build_persistence_lookup(test_df, target_col, window_days)
    preds = []
    for _, row in test_df.iterrows():
        key = (row["section_key"], pd.Timestamp(row[COL_DATE]))
        preds.append(lookup.get(key, np.nan))
    return np.array(preds)


def main():
    all_benchmark_rows = []
    all_persist_rows   = []
    all_common_rows    = []

    for target_name, target_col in TARGETS.items():
        for task in ["design", "monitoring"]:
            train_f = RESULTS_DIR / f"{task}_{target_name}_train.parquet"
            test_f  = RESULTS_DIR / f"{task}_{target_name}_test.parquet"
            if not train_f.exists() or not test_f.exists():
                continue

            train_df = pd.read_parquet(train_f)
            test_df  = pd.read_parquet(test_f)

            feature_cols = [c for c in train_df.columns
                            if c not in META_COLS
                            and c != target_col
                            and not c.startswith("LAG_")
                            and pd.api.types.is_numeric_dtype(train_df[c])]

            y_train = train_df[target_col].values
            y_test  = test_df[target_col].values
            label_prefix = f"{task}_{target_name}"

            # ── Benchmark 1: Mean predictor ───────────────────────────────────
            y_mean = np.full(len(y_test), np.nanmean(y_train))
            all_benchmark_rows.append(metrics(y_test, y_mean, f"mean_{label_prefix}"))

            # ── Benchmark 2: AASHTO exponential (IRI only) ────────────────────
            if target_name == "iri" and "AGE_YEARS" in train_df.columns:
                age_train = train_df["AGE_YEARS"].values
                age_test  = test_df["AGE_YEARS"].values
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        mask = ~(np.isnan(age_train) | np.isnan(y_train))
                        popt, _ = curve_fit(
                            aashto_func,
                            age_train[mask], y_train[mask],
                            p0=[1.0, 0.01], maxfev=5000,
                        )
                        y_aashto = aashto_func(age_test, *popt)
                        all_benchmark_rows.append(
                            metrics(y_test, y_aashto, f"aashto_{label_prefix}")
                        )
                    except Exception:
                        pass

            # ── Benchmark 3: Ridge regression ─────────────────────────────────
            X_train = train_df[feature_cols].fillna(0).values
            X_test  = test_df[feature_cols].fillna(0).values
            ridge_cv = GridSearchCV(
                Ridge(),
                {"alpha": np.logspace(-4, 4, 9)},
                scoring="neg_root_mean_squared_error",
                cv=5, n_jobs=-1,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ridge_cv.fit(X_train, y_train)
            y_ridge = ridge_cv.predict(X_test)
            all_benchmark_rows.append(
                metrics(y_test, y_ridge, f"ridge_{label_prefix}")
            )

            # ── Benchmark 4 + Common set: Persistence at k=1,2,3 years ────────
            # Common evaluation set: test obs with prior within 365d (most restrictive)
            persist_365 = persistence_predict(
                test_df, target_col, window_days=365
            )
            common_mask = ~np.isnan(persist_365)   # 251 obs for IRI design
            common_n    = common_mask.sum()

            y_test_common   = y_test[common_mask]

            # Monitoring model prediction on common set (if available)
            mon_lag_col = f"LAG_{target_col}"
            if task == "monitoring" and mon_lag_col in test_df.columns:
                y_mon_common = test_df[mon_lag_col].values[common_mask]
                all_common_rows.append(
                    metrics(y_test_common, y_mon_common,
                            f"monitoring_on_common_{target_name}")
                )

            for k in PERSISTENCE_YEARS:
                w = int(k * 365.25)
                y_persist = persistence_predict(test_df, target_col, w)
                # Full evaluation
                all_persist_rows.append(
                    metrics(y_test, y_persist, f"persist_k{k}_{label_prefix}")
                )
                # Common-set evaluation
                all_common_rows.append(
                    metrics(y_test_common, y_persist[common_mask],
                            f"persist_k{k}_common_{target_name}")
                )

            print(f"[{label_prefix}] benchmarks computed | common set n={common_n}")

    # Save outputs
    pd.DataFrame(all_benchmark_rows).to_csv(
        RESULTS_DIR / "benchmark_metrics.csv", index=False
    )
    pd.DataFrame(all_persist_rows).to_csv(
        RESULTS_DIR / "persistence_metrics.csv", index=False
    )
    pd.DataFrame(all_common_rows).to_csv(
        RESULTS_DIR / "common_set_comparison.csv", index=False
    )

    print("\nSaved → results/benchmark_metrics.csv")
    print("Saved → results/persistence_metrics.csv")
    print("Saved → results/common_set_comparison.csv")
    print("\n[09] Benchmarks complete.\n")


if __name__ == "__main__":
    main()
