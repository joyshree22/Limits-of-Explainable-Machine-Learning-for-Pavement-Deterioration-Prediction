"""
11_evaluate.py — Standard test-set evaluation with section-level bootstrap CIs.
Methodology §3.11.1–3.11.2: evaluate retrained model on 7 held-out test sections.
Bootstrap: 2000 iterations, section-level resampling, report as [lower, upper].
Outputs: results/test_metrics.csv   (R², RMSE, MAE + 95% CI)
         results/test_predictions_{arch}_{task}_{target}.parquet
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from config import (
    RESULTS_DIR, MODELS_DIR, TARGETS, BOOTSTRAP_N, BOOTSTRAP_CI, SEED,
    PERSISTENCE_YEARS, COL_DATE,
)

META_COLS = {
    "section_key", "STATE_CODE_EXP", "SHRP_ID", "PAVEMENT_FAMILY",
    "OBSERVATION_DATE", "CN_ASSIGN_DATE", "AGE_YEARS", "split",
}


def bootstrap_r2(y_true: np.ndarray, y_pred: np.ndarray,
                 section_keys: np.ndarray, n: int, rng) -> tuple[float, float]:
    unique_secs = np.unique(section_keys)
    r2_boot = []
    for _ in range(n):
        sampled = rng.choice(unique_secs, size=len(unique_secs), replace=True)
        boot_idx = np.concatenate([
            np.where(section_keys == sec)[0] for sec in sampled
        ])
        if boot_idx.size < 2:
            continue
        r2_boot.append(r2_score(y_true[boot_idx], y_pred[boot_idx]))
    lo, hi = np.percentile(r2_boot, BOOTSTRAP_CI)
    return round(lo, 4), round(hi, 4)


def evaluate(y_true, y_pred, section_keys, label, rng):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp, sk = y_true[mask], y_pred[mask], section_keys[mask]
    r2   = r2_score(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    mae  = mean_absolute_error(yt, yp)
    ci_lo, ci_hi = bootstrap_r2(yt, yp, sk, BOOTSTRAP_N, rng)
    return {
        "model": label,
        "n_obs": int(mask.sum()),
        "n_sections": int(np.unique(sk).shape[0]),
        "R2":    round(r2, 4),
        "R2_CI_lower": ci_lo,
        "R2_CI_upper": ci_hi,
        "RMSE":  round(rmse, 4),
        "MAE":   round(mae, 4),
    }


def persistence_predict(
    eval_df: pd.DataFrame,
    history_df: pd.DataFrame,
    target_col: str,
    window_days: int,
) -> np.ndarray:
    """Prior observed value within window_days from the same held-out section."""
    preds = np.full(len(eval_df), np.nan)
    eval_reset = eval_df.reset_index(drop=True)
    history = history_df.sort_values(["section_key", COL_DATE])
    for sec, hist_grp in history.groupby("section_key", sort=False):
        eval_grp = eval_reset[eval_reset["section_key"] == sec]
        if eval_grp.empty:
            continue
        hist_grp = hist_grp.sort_values(COL_DATE)
        hist_dates = pd.to_datetime(hist_grp[COL_DATE]).values
        hist_vals = hist_grp[target_col].values
        for eval_idx, row in eval_grp.iterrows():
            cur = pd.Timestamp(row[COL_DATE])
            prior_mask = (hist_dates < cur) & (hist_dates >= cur - pd.Timedelta(days=window_days))
            if prior_mask.any():
                preds[eval_idx] = hist_vals[np.where(prior_mask)[0][-1]]
    return preds


def metric_row(y_true, y_pred, label: str) -> dict | None:
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 2:
        return None
    yt, yp = y_true[mask], y_pred[mask]
    return {
        "model": label,
        "n": int(mask.sum()),
        "R2": round(r2_score(yt, yp), 4),
        "RMSE": round(np.sqrt(mean_squared_error(yt, yp)), 4),
        "MAE": round(mean_absolute_error(yt, yp), 4),
    }


def write_common_set_comparisons(rng):
    """
    Compare trained monitoring models and persistence benchmarks on identical
    section/date rows. This intentionally runs after model training.
    """
    rows = []
    pred_rows = []

    for target_name, target_col in TARGETS.items():
        test_f = RESULTS_DIR / f"monitoring_{target_name}_test.parquet"
        if not test_f.exists():
            continue
        test_df = pd.read_parquet(test_f).reset_index(drop=True)
        history_f = RESULTS_DIR / f"design_{target_name}_test.parquet"
        history_df = pd.read_parquet(history_f) if history_f.exists() else test_df
        y_true = test_df[target_col].values

        persistence = {}
        for k in PERSISTENCE_YEARS:
            persistence[f"persist_k{k}"] = persistence_predict(
                test_df, history_df, target_col, int(k * 365.25)
            )

        common_mask = np.ones(len(test_df), dtype=bool)
        for pred in persistence.values():
            common_mask &= ~np.isnan(pred)

        for k, pred in persistence.items():
            row = metric_row(y_true[common_mask], pred[common_mask], f"{k}_common_{target_name}")
            if row:
                rows.append(row)

        for arch in ["xgb", "rf"]:
            model_f = MODELS_DIR / f"{arch}_monitoring_{target_name}.joblib"
            if not model_f.exists():
                continue
            model, feature_cols = joblib.load(model_f)
            y_pred = model.predict(test_df[feature_cols].fillna(0).values)
            row = metric_row(
                y_true[common_mask], y_pred[common_mask],
                f"{arch}_monitoring_common_{target_name}"
            )
            if row:
                rows.append(row)

            for i in np.where(common_mask)[0]:
                pred_rows.append({
                    "target": target_name,
                    "model": f"{arch}_monitoring",
                    "section_key": test_df.loc[i, "section_key"],
                    "OBSERVATION_DATE": test_df.loc[i, COL_DATE],
                    "actual": y_true[i],
                    "predicted": y_pred[i],
                })

        # Delta monitoring is a sensitivity analysis for IRI only. It is scored
        # on the same common rows as absolute monitoring/persistence.
        if target_name == "iri":
            delta_col = f"DELTA_{target_col}"
            delta_f = RESULTS_DIR / f"monitoring_delta_{target_name}_test.parquet"
            delta_df = pd.read_parquet(delta_f).reset_index(drop=True) if delta_f.exists() else None
            for arch in ["xgb", "rf"]:
                model_f = MODELS_DIR / f"{arch}_monitoring_delta_{target_name}.joblib"
                if (
                    not model_f.exists()
                    or delta_df is None
                    or delta_col not in delta_df.columns
                    or len(delta_df) != len(test_df)
                ):
                    continue
                model, feature_cols = joblib.load(model_f)
                delta_pred = model.predict(delta_df[feature_cols].fillna(0).values)
                abs_pred = delta_pred + delta_df[f"LAG_{target_col}"].values
                row = metric_row(
                    y_true[common_mask], abs_pred[common_mask],
                    f"{arch}_monitoring_delta_common_{target_name}"
                )
                if row:
                    persist_rmse = next(
                        (r["RMSE"] for r in rows
                         if r["model"] == f"persist_k1_common_{target_name}"),
                        np.nan,
                    )
                    row["skill_vs_persist_k1"] = (
                        round(1.0 - row["RMSE"] / persist_rmse, 4)
                        if persist_rmse and not np.isnan(persist_rmse) else np.nan
                    )
                    rows.append(row)

                for i in np.where(common_mask)[0]:
                    pred_rows.append({
                        "target": target_name,
                        "model": f"{arch}_monitoring_delta",
                        "section_key": test_df.loc[i, "section_key"],
                        "OBSERVATION_DATE": test_df.loc[i, COL_DATE],
                        "actual": y_true[i],
                        "predicted": abs_pred[i],
                    })

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "common_set_comparison.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(RESULTS_DIR / "common_set_predictions.csv", index=False)
    print("Saved → results/common_set_comparison.csv")
    print("Saved → results/common_set_predictions.csv")


def main():
    rng = np.random.default_rng(SEED)
    rows = []

    for target_name, target_col in TARGETS.items():
        for task in ["design", "monitoring"]:
            test_f = RESULTS_DIR / f"{task}_{target_name}_test.parquet"
            if not test_f.exists():
                continue

            test_df = pd.read_parquet(test_f)
            y_test  = test_df[target_col].values
            sk      = test_df["section_key"].values

            for arch in ["xgb", "rf"]:
                model_f = MODELS_DIR / f"{arch}_{task}_{target_name}.joblib"
                if not model_f.exists():
                    print(f"  Model not found: {model_f.name}")
                    continue

                model, feature_cols = joblib.load(model_f)
                X_test = test_df[feature_cols].fillna(0).values
                y_pred = model.predict(X_test)

                label = f"{arch}_{task}_{target_name}"
                result = evaluate(y_test, y_pred, sk, label, rng)
                rows.append(result)

                # Save predictions for residual analysis
                pred_df = test_df[["section_key", "STATE_CODE_EXP", "OBSERVATION_DATE",
                                   target_col]].copy()
                pred_df["predicted"] = y_pred
                pred_df["residual"]  = y_test - y_pred
                pred_df.to_parquet(
                    RESULTS_DIR / f"test_predictions_{label}.parquet", index=False
                )

                print(f"  {label}: R²={result['R2']} "
                      f"[{result['R2_CI_lower']}, {result['R2_CI_upper']}]  "
                      f"RMSE={result['RMSE']}")

    out = RESULTS_DIR / "test_metrics.csv"
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(out, index=False)
    if not metrics_df.empty:
        metrics_df[metrics_df["model"].str.contains("_design_")].to_csv(
            RESULTS_DIR / "design_model_metrics.csv", index=False
        )
        metrics_df[metrics_df["model"].str.contains("_monitoring_")].to_csv(
            RESULTS_DIR / "monitoring_model_metrics.csv", index=False
        )
    print(f"\nSaved → {out}")
    write_common_set_comparisons(rng)
    print("\n[11] Evaluation complete.\n")


if __name__ == "__main__":
    main()
