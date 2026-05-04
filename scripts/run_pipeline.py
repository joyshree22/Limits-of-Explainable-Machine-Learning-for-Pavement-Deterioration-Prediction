"""
run_pipeline.py — Master script: runs all pipeline steps in order.
Usage:  python run_pipeline.py
        python run_pipeline.py --steps 01 02 03   (run specific steps only)
        python run_pipeline.py --from 07           (resume from a step)
Each step is a separate module; stdout from each is captured and timed.
"""

import sys
import time
import argparse
import importlib.util
import traceback
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))

PIPELINE = [
    ("01", "01_load_audit",   "Load raw data and produce missingness audit"),
    ("02", "02_aggregate",    "Visit-level aggregation (IRI: 3,233 → 650 obs)"),
    ("03", "03_features",     "Feature engineering + 9 physics-based composites"),
    ("04", "04_missingness",  "Apply 50% missingness threshold per target"),
    ("05", "05_partition",    "Section-wise 34/7/7 train/val/test split"),
    ("06", "06_impute_scale", "Group-median imputation + StandardScaler"),
    ("07", "07_collinearity", "Spearman hierarchical clustering → 114–118 features"),
    ("08", "08_tasks",        "Construct design and monitoring datasets"),
    ("09", "09_benchmarks",   "Four benchmarks + persistence k=1,2,3"),
    ("10", "10_train",        "XGBoost + RF training with Optuna (longest step)"),
    ("11", "11_evaluate",     "Standard test-set evaluation + bootstrap CIs"),
    ("12", "12_loo",          "Leave-One-Region-Out validation"),
    ("13", "13_shap",         "SHAP TreeExplainer + cross-model consistency"),
    ("14", "14_pdp",          "Partial dependence plots"),
    ("15", "15_residuals",    "Residual diagnostic plots"),
]


def run_step(step_id: str, module_name: str, description: str) -> bool:
    print(f"\n{'='*65}")
    print(f"  STEP {step_id}: {description}")
    print(f"{'='*65}")
    t0 = time.time()
    try:
        # Use spec_from_file_location so numeric-prefixed filenames work
        file_path = SCRIPTS_DIR / f"{module_name}.py"
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.main()
        elapsed = time.time() - t0
        print(f"  ✓ Step {step_id} completed in {elapsed:.1f}s")
        return True
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  ✗ Step {step_id} FAILED after {elapsed:.1f}s")
        print(f"  Error: {e}")
        traceback.print_exc()
        return False


def parse_args():
    parser = argparse.ArgumentParser(description="LTPP ML pipeline runner")
    parser.add_argument("--steps", nargs="*", help="Run only specific steps: 01 03 05")
    parser.add_argument("--from",  dest="from_step", help="Resume from step: 07")
    return parser.parse_args()


def main():
    args = parse_args()
    t_total = time.time()

    steps_to_run = PIPELINE
    if args.steps:
        selected = set(args.steps)
        steps_to_run = [(sid, mod, desc) for sid, mod, desc in PIPELINE
                        if sid in selected]
    elif args.from_step:
        from_idx = next((i for i, (sid, _, _) in enumerate(PIPELINE)
                         if sid == args.from_step), None)
        if from_idx is None:
            print(f"Step {args.from_step} not found.")
            sys.exit(1)
        steps_to_run = PIPELINE[from_idx:]

    results = {}
    for step_id, module_name, description in steps_to_run:
        ok = run_step(step_id, module_name, description)
        results[step_id] = "PASS" if ok else "FAIL"
        if not ok:
            print(f"\nPipeline halted at step {step_id}. Fix the error and resume with:")
            print(f"  python run_pipeline.py --from {step_id}")
            break

    total_elapsed = time.time() - t_total
    print(f"\n{'='*65}")
    print(f"  PIPELINE SUMMARY  ({total_elapsed/60:.1f} min total)")
    print(f"{'='*65}")
    for step_id, status in results.items():
        mark = "✓" if status == "PASS" else "✗"
        desc = next(d for s, _, d in PIPELINE if s == step_id)
        print(f"  {mark} Step {step_id}: {desc}")


if __name__ == "__main__":
    main()
