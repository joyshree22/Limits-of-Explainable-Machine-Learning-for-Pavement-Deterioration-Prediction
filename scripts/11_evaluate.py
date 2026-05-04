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
        mask = np.isin(section_keys, sampled)
        if mask.sum() < 2:
            continue
        r2_boot.append(r2_score(y_true[mask], y_pred[mask]))
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
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved → {out}")
    print("\n[11] Evaluation complete.\n")


if __name__ == "__main__":
    main()
