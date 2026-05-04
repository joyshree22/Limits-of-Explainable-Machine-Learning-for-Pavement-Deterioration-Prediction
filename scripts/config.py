"""
config.py — Central configuration for the analysis pipeline.
All paths, constants, and random seeds are defined here.
Every pipeline script imports from this module.
"""

from pathlib import Path

# ── Directories ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
MODELS_DIR  = ROOT / "models"

for d in [RESULTS_DIR, FIGURES_DIR, MODELS_DIR]:
    d.mkdir(exist_ok=True)

# ── Source file ───────────────────────────────────────────────────────────────
RAW_CSV = DATA_DIR / "master-data-clean.csv"

# ── Identifiers ───────────────────────────────────────────────────────────────
COL_REGION  = "STATE_CODE_EXP"
COL_SECTION = "SHRP_ID"
COL_DATE    = "OBSERVATION_DATE"
COL_FAMILY  = "PAVEMENT_FAMILY"
COL_CN_DATE = "CN_ASSIGN_DATE"

# ── Target columns ────────────────────────────────────────────────────────────
TARGETS = {
    "iri":      "IRI_MRI",
    "rutting":  "RUT_LLH_DEPTH_1_8_MEAN",
    "distress": "DIS_HPMS16_CRACKING_PERCENT_AC",
}

# ── Regions and section counts ────────────────────────────────────────────────
REGIONS = ["Arizona", "Georgia", "Ohio", "Ontario"]

REGION_SPLIT = {
    # region: (n_train, n_val, n_test)
    "Arizona": (16, 3, 3),
    "Georgia": (8, 2, 2),
    "Ohio":    (5, 1, 1),
    "Ontario": (5, 1, 1),
}

# ── Pavement families ─────────────────────────────────────────────────────────
PAVEMENT_FAMILIES = ["ACUB", "ACATB", "ACTB"]

# ── Missingness threshold ─────────────────────────────────────────────────────
MISS_THRESHOLD = 0.50

# ── Collinearity threshold ────────────────────────────────────────────────────
SPEARMAN_CORR_THRESHOLD = 0.85   # |ρ| > 0.85 → features are in same cluster

# ── Monitoring lag windows (days) ─────────────────────────────────────────────
LAG_WINDOWS = {
    "iri":      730,
    "rutting":  730,
    "distress": 1095,
}

# Monitoring viability floor
MONITOR_MIN_OBS      = 150
MONITOR_MIN_SECTIONS = 20

# ── Persistence benchmark horizons ───────────────────────────────────────────
PERSISTENCE_YEARS = [1, 2, 3]   # → 365, 730, 1095 days

# ── Optuna / training ─────────────────────────────────────────────────────────
OPTUNA_TRIALS = {"iri": 150, "rutting": 100, "distress": 100}
OPTUNA_EARLY_STOP_PATIENCE = 30     # trials with < 0.1% improvement → stop
OPTUNA_EARLY_STOP_TOL      = 0.001  # 0.1%
CV_N_SPLITS = 5

# ── Bootstrap ─────────────────────────────────────────────────────────────────
BOOTSTRAP_N = 2000
BOOTSTRAP_CI = (2.5, 97.5)

# ── SHAP stability threshold (pre-specified) ──────────────────────────────────
SHAP_CONSISTENCY_THRESHOLD = 0.75

# ── Random seed ───────────────────────────────────────────────────────────────
SEED = 42

# ── Ontario sub-group FI reference ───────────────────────────────────────────
ONTARIO_SECTIONS = {
    "1620": "moderate", "1680": "moderate", "1806": "moderate",
    "1622": "high",
    "903":  "extreme",  "960":  "extreme",  "961":  "extreme",
}
ONTARIO_FI_GROUPS = ONTARIO_SECTIONS   # alias used by 12_loo.py

# Mean CLIM_FREEZE_INDEX per Ontario section (computed from raw data; used in 12_loo.py
# because the design parquets carry only scaled features and FI was collinearity-excluded)
ONTARIO_SECTION_FI = {
    "903":  1225.6, "960":  1222.1, "961":  1225.6,
    "1620": 604.6,  "1622": 863.7,  "1680": 600.9,  "1806": 598.4,
}
