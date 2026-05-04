"""
10_train.py — XGBoost and Random Forest training with Optuna TPE + GroupKFold.
Methodology §3.10: GroupKFold(5) by section_key prevents section-level leakage.
Regional inverse-frequency sample weights counter Arizona's overrepresentation.
Final model retrained on 41-section (train+val) set with validation-selected HPs unchanged.
Outputs: models/{arch}_{task}_{target}.joblib
         results/best_params_{task}_{target}.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import warnings
import numpy as np
import pandas as pd
import joblib
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from config import (
    RESULTS_DIR, MODELS_DIR, TARGETS, COL_REGION,
    OPTUNA_TRIALS, OPTUNA_EARLY_STOP_PATIENCE, OPTUNA_EARLY_STOP_TOL,
    CV_N_SPLITS, SEED,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

META_COLS = {
    "section_key", "STATE_CODE_EXP", "SHRP_ID", "PAVEMENT_FAMILY",
    "OBSERVATION_DATE", "CN_ASSIGN_DATE", "AGE_YEARS", "split",
}


def compute_sample_weights(df: pd.DataFrame) -> np.ndarray:
    region_counts = df[COL_REGION].value_counts()
    w = df[COL_REGION].map(lambda r: 1.0 / region_counts[r])
    w = w / w.sum() * len(w)
    return w.values


def get_feature_cols(df: pd.DataFrame, target_col: str) -> list[str]:
    return [c for c in df.columns
            if c not in META_COLS
            and c != target_col
            and pd.api.types.is_numeric_dtype(df[c])]


def make_xgb_objective(X, y, groups, weights, n_splits):
    gkf = GroupKFold(n_splits=n_splits)
    best_scores = []

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 50, 500),
            "max_depth":         trial.suggest_int("max_depth", 3, 12),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state":      SEED,
            "n_jobs":            -1,
            "verbosity":         0,
        }
        model = XGBRegressor(**params)
        rmse_scores = []
        for tr_idx, val_idx in gkf.split(X, y, groups):
            Xtr, Xval = X[tr_idx], X[val_idx]
            ytr, yval = y[tr_idx], y[val_idx]
            wtr       = weights[tr_idx]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(Xtr, ytr, sample_weight=wtr, eval_set=[(Xval, yval)],
                          verbose=False)
            pred = model.predict(Xval)
            rmse_scores.append(np.sqrt(mean_squared_error(yval, pred)))
        score = np.mean(rmse_scores)
        best_scores.append(score)

        # Early stopping: check last patience trials
        if len(best_scores) >= OPTUNA_EARLY_STOP_PATIENCE:
            recent = best_scores[-OPTUNA_EARLY_STOP_PATIENCE:]
            if (recent[0] - min(recent)) / (abs(recent[0]) + 1e-9) < OPTUNA_EARLY_STOP_TOL:
                trial.study.stop()
        return score

    return objective


def make_rf_objective(X, y, groups, weights, n_splits):
    gkf = GroupKFold(n_splits=n_splits)
    best_scores = []

    def objective(trial):
        params = {
            "n_estimators":    trial.suggest_int("n_estimators", 50, 500),
            "max_depth":       trial.suggest_int("max_depth", 3, 12),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features":    trial.suggest_categorical("max_features",
                                   ["sqrt", "log2", 0.3, 0.5, 0.7, 0.9]),
            "random_state": SEED,
            "n_jobs": -1,
        }
        model = RandomForestRegressor(**params)
        rmse_scores = []
        for tr_idx, val_idx in gkf.split(X, y, groups):
            Xtr, Xval = X[tr_idx], X[val_idx]
            ytr, yval = y[tr_idx], y[val_idx]
            wtr       = weights[tr_idx]
            model.fit(Xtr, ytr, sample_weight=wtr)
            pred = model.predict(Xval)
            rmse_scores.append(np.sqrt(mean_squared_error(yval, pred)))
        score = np.mean(rmse_scores)
        best_scores.append(score)

        if len(best_scores) >= OPTUNA_EARLY_STOP_PATIENCE:
            recent = best_scores[-OPTUNA_EARLY_STOP_PATIENCE:]
            if (recent[0] - min(recent)) / (abs(recent[0]) + 1e-9) < OPTUNA_EARLY_STOP_TOL:
                trial.study.stop()
        return score

    return objective


# Fault 4 fix: rutting mechanism regime separation
WARM_REGIONS   = {"Arizona", "Georgia"}
FREEZE_REGIONS = {"Ohio", "Ontario"}


def train_one_model(arch, task_label, target_name, target_col,
                    train_df, val_df, model_suffix="", actual_target=None):
    """Train a single model variant. actual_target overrides target_col (used for ΔIRI)."""
    fit_col = actual_target if actual_target else target_col

    final_df = pd.concat([train_df, val_df], ignore_index=True) if len(val_df) else train_df.copy()

    feature_cols = get_feature_cols(train_df, fit_col)

    valid_mask   = train_df[fit_col].notna()
    X_train = train_df.loc[valid_mask, feature_cols].fillna(0).values
    y_train = train_df.loc[valid_mask, fit_col].values
    groups  = train_df.loc[valid_mask, "section_key"].values
    weights = compute_sample_weights(train_df[valid_mask])

    X_final = final_df[feature_cols].fillna(0).values
    y_final = final_df[fit_col].fillna(0).values
    w_final = compute_sample_weights(final_df)

    if len(X_train) < 10:
        print(f"  SKIP {arch} {task_label}_{target_name}{model_suffix} — insufficient data (n={len(X_train)})")
        return

    n_trials = OPTUNA_TRIALS.get(target_name, 100)
    print(f"  Tuning {arch.upper()} | {task_label}_{target_name}{model_suffix} | {n_trials} trials | n={len(X_train)} ...")

    objective = (make_xgb_objective if arch == "xgb" else make_rf_objective)(
        X_train, y_train, groups, weights, CV_N_SPLITS
    )
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = study.best_params

    param_path = RESULTS_DIR / f"best_params_{arch}_{task_label}_{target_name}{model_suffix}.json"
    with open(param_path, "w") as f:
        json.dump(best_params, f, indent=2)

    if arch == "xgb":
        final_model = XGBRegressor(**best_params, random_state=SEED, n_jobs=-1, verbosity=0)
    else:
        final_model = RandomForestRegressor(**best_params, random_state=SEED, n_jobs=-1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_model.fit(X_final, y_final, sample_weight=w_final)

    model_path = MODELS_DIR / f"{arch}_{task_label}_{target_name}{model_suffix}.joblib"
    joblib.dump((final_model, feature_cols), model_path)
    print(f"    Saved → {model_path.name} | best CV RMSE={study.best_value:.4f}")


def train_target_task(target_name: str, task: str):
    target_col = TARGETS[target_name]

    train_f = RESULTS_DIR / f"{task}_{target_name}_train.parquet"
    val_f   = RESULTS_DIR / f"{task}_{target_name}_val.parquet"
    if not train_f.exists():
        print(f"  [{task}_{target_name}] training data not found — skip")
        return

    train_df = pd.read_parquet(train_f)
    val_df   = pd.read_parquet(val_f) if val_f.exists() else pd.DataFrame()

    for arch in ["xgb", "rf"]:
        # Global model (all regions)
        train_one_model(arch, task, target_name, target_col, train_df, val_df)

        # Fault 4: separate regime rutting models
        if target_name == "rutting" and task == "design":
            for regime, regions in [("warm", WARM_REGIONS), ("freeze", FREEZE_REGIONS)]:
                tr_r = train_df[train_df[COL_REGION].isin(regions)]
                vl_r = val_df[val_df[COL_REGION].isin(regions)] if len(val_df) else pd.DataFrame()
                train_one_model(arch, task, target_name, target_col,
                                tr_r, vl_r, model_suffix=f"_{regime}")

        # Fault 2: ΔIRI skill-score monitoring model
        if target_name == "iri" and task == "monitoring":
            delta_col = f"DELTA_{target_col}"
            delta_f   = RESULTS_DIR / f"monitoring_delta_{target_name}_train.parquet"
            if delta_f.exists():
                d_train = pd.read_parquet(delta_f)
                d_val_f = RESULTS_DIR / f"monitoring_delta_{target_name}_val.parquet"
                d_val   = pd.read_parquet(d_val_f) if d_val_f.exists() else pd.DataFrame()
                if delta_col in d_train.columns and d_train[delta_col].notna().sum() >= 20:
                    train_one_model(arch, "monitoring_delta", target_name, target_col,
                                    d_train, d_val, actual_target=delta_col)


def main():
    for target_name in TARGETS:
        for task in ["design", "monitoring"]:
            print(f"\n{'='*55}")
            print(f" Target: {target_name}  |  Task: {task}")
            print(f"{'='*55}")
            train_target_task(target_name, task)

    print("\n[10] Model training complete.\n")


if __name__ == "__main__":
    main()
