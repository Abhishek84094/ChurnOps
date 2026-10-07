"""
Phase 15 — MLflow Tracking
==========================
Registers and logs all experiment runs, hyperparameters, cross-validation metrics,
generalization gaps, holdout test metrics, and diagnostic artifacts into MLflow.

Architecture:
  - Tracking URI: sqlite:///mlruns/mlflow.db
  - Experiment Name: ChurnOps_Model_Comparison
  - Runs Logged:
      1. Baseline DummyClassifier
      2. Baseline LogisticRegression
      3. Baseline RandomForest
      4. Baseline XGBoost
      5. Tuned LogisticRegression
      6. Tuned RandomForest
      7. Tuned XGBoost
      8. Ensemble VotingClassifier
      9. Ensemble StackingClassifier
      10. Production Champion (Tuned XGBoost with holdout test metrics)
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import mlflow
import pandas as pd

from src.utils.config import PARAMS
from src.utils.logging import get_logger

log = get_logger("mlflow_track")

TRACKING_URI: str = PARAMS["mlflow"]["tracking_uri"]
EXPERIMENT_NAME: str = PARAMS["mlflow"]["experiment_name"]
REPORTS_DIR = Path("reports")
MLRUNS_DIR = Path("mlruns")


def setup_mlflow() -> str:
    """Configures MLflow tracking URI and ensures experiment exists with valid file URI artifact location."""
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    artifacts_dir = (MLRUNS_DIR / "artifacts").resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    artifact_uri = artifacts_dir.as_uri()

    mlflow.set_tracking_uri(TRACKING_URI)
    log.info("MLflow tracking URI set to: %s", TRACKING_URI)

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is not None and not experiment.artifact_location.startswith("file:"):
        # Recreate experiment with proper file URI
        mlflow.delete_experiment(experiment.experiment_id)
        experiment = None

    if experiment is None:
        # Create experiment with explicit file:/// URI
        import uuid
        exp_name = EXPERIMENT_NAME
        try:
            exp_id = mlflow.create_experiment(
                name=exp_name,
                artifact_location=artifact_uri,
            )
        except Exception:
            # If name was soft-deleted, use a unique name
            exp_name = f"{EXPERIMENT_NAME}_{int(time.time())}"
            exp_id = mlflow.create_experiment(
                name=exp_name,
                artifact_location=artifact_uri,
            )
        log.info("Created new MLflow experiment '%s' (ID: %s)", exp_name, exp_id)
    else:
        exp_id = experiment.experiment_id
        log.info("Using existing MLflow experiment '%s' (ID: %s)", experiment.name, exp_id)

    mlflow.set_experiment(experiment_id=exp_id)
    return exp_id



def log_all_runs(exp_id: str) -> Dict[str, Any]:
    """Reads all historical phase reports and logs standardized MLflow runs."""
    log.info("\nGathering reports from Phase 9, 10, 12, 13, and 14...")

    # Load reports
    with open(REPORTS_DIR / "baseline_summary.json", "r") as f:
        base_rep = json.load(f)
    with open(REPORTS_DIR / "cv_analysis_report.json", "r") as f:
        cv_rep = json.load(f)
    with open(REPORTS_DIR / "tuning_report.json", "r") as f:
        tune_rep = json.load(f)
    with open(REPORTS_DIR / "improvement_report.json", "r") as f:
        imp_rep = json.load(f)
    with open(REPORTS_DIR / "final_model_evaluation.json", "r") as f:
        final_rep = json.load(f)

    logged_runs: List[Dict[str, Any]] = []

    # 1. Baseline Runs
    base_bv = cv_rep.get("bias_variance", {})
    models_summary = base_rep.get("summary", {})
    for model_name, stats in models_summary.items():
        run_name = f"Baseline_{model_name}"
        log.info("Logging run: %s", run_name)
        gap = base_bv.get(model_name, {}).get("gap", 0.0)

        # Parse string metrics "0.6258 +/- 0.0257"
        f1_val = float(stats["f1"].split()[0]) if "f1" in stats else 0.0
        auc_val = float(stats["roc_auc"].split()[0]) if "roc_auc" in stats else 0.0
        rec_val = float(stats["recall"].split()[0]) if "recall" in stats else 0.0
        prec_val = float(stats["precision"].split()[0]) if "precision" in stats else 0.0
        acc_val = float(stats["accuracy"].split()[0]) if "accuracy" in stats else 0.0

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tags({
                "model_stage": "baseline",
                "model_family": model_name,
                "phase": "09_baseline",
                "is_champion": "false",
            })
            mlflow.log_params({
                "model_type": model_name,
                "tuning": "none (default)",
                "cv_folds": 5,
                "random_state": 42,
            })
            mlflow.log_metrics({
                "val_f1": f1_val,
                "val_roc_auc": auc_val,
                "val_recall": rec_val,
                "val_precision": prec_val,
                "val_accuracy": acc_val,
                "generalization_gap": gap,
            })
            logged_runs.append({
                "run_id": run.info.run_id,
                "run_name": run_name,
                "val_f1": f1_val,
                "val_roc_auc": auc_val,
                "gap": gap,
            })


    # 2. Tuned Runs (Phase 12)
    for model_name, t_data in tune_rep["tuned_models"].items():
        run_name = f"Tuned_{model_name}"
        log.info("Logging run: %s", run_name)
        m = t_data["metrics"]

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tags({
                "model_stage": "tuned",
                "model_family": model_name,
                "phase": "12_hpo",
                "is_champion": "false",
            })
            # Log params
            params_to_log = {f"tuned__{k}": str(v) for k, v in t_data["best_params"].items()}
            params_to_log["model_type"] = model_name
            params_to_log["cv_folds"] = 5
            mlflow.log_params(params_to_log)

            mlflow.log_metrics({
                "val_f1": m["mean_val_f1"],
                "val_f1_std": m["std_val_f1"],
                "train_f1": m["mean_train_f1"],
                "val_roc_auc": m["mean_roc_auc"],
                "val_recall": m["mean_recall"],
                "val_precision": m["mean_precision"],
                "val_accuracy": m["mean_accuracy"],
                "generalization_gap": m["generalization_gap"],
            })
            if (REPORTS_DIR / "tuning_comparison.png").exists():
                mlflow.log_artifact(str(REPORTS_DIR / "tuning_comparison.png"))

            logged_runs.append({
                "run_id": run.info.run_id,
                "run_name": run_name,
                "val_f1": m["mean_val_f1"],
                "val_roc_auc": m["mean_roc_auc"],
                "gap": m["generalization_gap"],
            })

    # 3. Ensemble Runs (Phase 13)
    for ens_name in ["VotingClassifier", "StackingClassifier"]:
        if ens_name in imp_rep["models"]:
            m = imp_rep["models"][ens_name]
            run_name = f"Ensemble_{ens_name}"
            log.info("Logging run: %s", run_name)

            with mlflow.start_run(run_name=run_name) as run:
                mlflow.set_tags({
                    "model_stage": "ensemble",
                    "model_family": ens_name,
                    "phase": "13_improvement",
                    "is_champion": "false",
                })
                mlflow.log_params({
                    "ensemble_type": ens_name,
                    "base_models": "LR, RF, XGB",
                    "optimal_threshold": m["optimal_threshold"],
                })
                mlflow.log_metrics({
                    "val_f1_default": m["mean_val_f1"],
                    "val_f1_optimal": m["f1_at_optimal_threshold"],
                    "val_roc_auc": m["mean_roc_auc"],
                    "val_recall_optimal": m["recall_at_optimal_threshold"],
                    "val_precision_optimal": m["precision_at_optimal_threshold"],
                    "generalization_gap": m["generalization_gap"],
                })
                if (REPORTS_DIR / "ensemble_comparison.png").exists():
                    mlflow.log_artifact(str(REPORTS_DIR / "ensemble_comparison.png"))

                logged_runs.append({
                    "run_id": run.info.run_id,
                    "run_name": run_name,
                    "val_f1": m["mean_val_f1"],
                    "val_roc_auc": m["mean_roc_auc"],
                    "gap": m["generalization_gap"],
                })

    # 4. Production Champion Run (Phase 14)
    champ_name = final_rep["champion_model"]
    run_name = f"Champion_{champ_name}"
    log.info("Logging run: %s (CHAMPION)", run_name)
    test_res = final_rep["holdout_test_results"]
    m_def = test_res["default_threshold_metrics"]
    m_opt = test_res["optimal_threshold_metrics"]

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({
            "model_stage": "production_champion",
            "model_family": champ_name,
            "phase": "14_selection",
            "is_champion": "true",
            "status": "ready_for_registry",
        })
        mlflow.log_params({
            "model_name": champ_name,
            "optimal_threshold": final_rep["optimal_threshold"],
            "cv_val_f1": 0.6371,
            "cv_roc_auc": 0.8493,
            "cv_gap": 0.0194,
        })
        mlflow.log_metrics({
            "test_roc_auc": test_res["roc_auc"],
            "test_f1_default": m_def["f1"],
            "test_recall_default": m_def["recall"],
            "test_precision_default": m_def["precision"],
            "test_accuracy_default": m_def["accuracy"],
            "test_f1_optimal": m_opt["f1"],
            "test_recall_optimal": m_opt["recall"],
            "test_precision_optimal": m_opt["precision"],
            "test_accuracy_optimal": m_opt["accuracy"],
        })
        # Log holdout test artifacts
        for art in ["test_confusion_matrix.png", "test_roc_pr_curves.png"]:
            if (REPORTS_DIR / art).exists():
                mlflow.log_artifact(str(REPORTS_DIR / art))

        logged_runs.append({
            "run_id": run.info.run_id,
            "run_name": run_name,
            "test_f1": m_def["f1"],
            "test_roc_auc": test_res["roc_auc"],
            "test_recall": m_def["recall"],
            "is_champion": True,
        })

    log.info("All %d runs logged successfully into MLflow.", len(logged_runs))
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": 15,
        "experiment_name": EXPERIMENT_NAME,
        "experiment_id": exp_id,
        "tracking_uri": TRACKING_URI,
        "total_runs": len(logged_runs),
        "logged_runs": logged_runs,
    }


def main():
    t0 = time.perf_counter()
    exp_id = setup_mlflow()
    summary = log_all_runs(exp_id)
    elapsed = time.perf_counter() - t0

    out_path = REPORTS_DIR / "mlflow_runs_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info("Saved summary -> %s", out_path)

    print("\n" + "=" * 80)
    print(f"PHASE 15 — MLFLOW TRACKING COMPLETE  |  Completed in {elapsed:.2f}s")
    print("=" * 80)
    print(f"Tracking URI:    {summary['tracking_uri']}")
    print(f"Experiment Name: {summary['experiment_name']} (ID: {summary['experiment_id']})")
    print(f"Total Runs Logged: {summary['total_runs']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
