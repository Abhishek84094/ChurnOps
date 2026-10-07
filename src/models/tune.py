"""
Phase 12 — Hyperparameter Optimization (HPO)
============================================
Performs systematic hyperparameter optimization for LogisticRegression,
RandomForest, and XGBoost using RandomizedSearchCV across the exact 5-fold CV
partitions from Phase 9/10/11.

Key Objectives:
  1. Regularize tree models (RF & XGB) to reduce the ~0.40 generalization gap
     below the strict 0.08 gate threshold.
  2. Optimize validation F1, ROC-AUC, and Recall simultaneously.
  3. Verify registry gates on tuned cross-validation performance.
  4. Fit the best tuned pipelines on the entire training set (train.csv)
     and persist to models/tuned_<model>.joblib.
  5. Generate visual comparison (Baseline vs. Tuned F1 and Gap) and save
     reports/tuning_report.json and reports/tuning_comparison.png.

Leakage Rule (R3):
  Strictly executed on data/processed/train.csv and cv_folds.json.
  Locked test set (data/processed/test.csv) is never accessed.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

from src.data.preprocessing import build_full_pipeline
from src.utils.config import PARAMS
from src.utils.logging import get_logger

log = get_logger("tune")

RANDOM_STATE: int = PARAMS["random_state"]
TARGET: str = PARAMS["dataset"]["target_column"]
GAP_THRESHOLD: float = PARAMS["diagnosis"]["gap_threshold"]
REGISTRY_GATES: Dict[str, float] = PARAMS["registry"]
REPORTS_DIR = Path("reports")
MODELS_DIR = Path("models")


def _compute_spw(y: pd.Series) -> float:
    neg = int((y == 0).sum())
    pos = int((y == 1).sum())
    return float(neg / pos) if pos > 0 else 1.0


def _load_data_and_folds() -> Tuple[pd.DataFrame, pd.Series, List[Tuple[List[int], List[int]]]]:
    train_path = Path("data/processed/train.csv")
    folds_path = Path("data/processed/cv_folds.json")
    if not train_path.exists():
        raise FileNotFoundError(f"Missing {train_path}")
    if not folds_path.exists():
        raise FileNotFoundError(f"Missing {folds_path}")

    df_train = pd.read_csv(train_path)
    with open(folds_path, "r", encoding="utf-8") as f:
        cv_folds = json.load(f)

    X = df_train.drop(columns=[TARGET])
    y = df_train[TARGET]

    cv_splits = [
        (cv_folds[k]["train_indices"], cv_folds[k]["val_indices"])
        for k in sorted(cv_folds.keys())
    ]
    return X, y, cv_splits


def get_search_spaces() -> Dict[str, Dict[str, Any]]:
    """Returns candidate parameter distributions derived from Phase 11 diagnosis."""
    return {
        "LogisticRegression": {
            "estimator": LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
            ),
            "scale_numeric": True,
            "n_iter": 12,
            "param_distributions": {
                "model__C": [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
                "model__solver": ["lbfgs", "saga"],
                "model__class_weight": ["balanced"],
            },
        },
        "RandomForest": {
            "estimator": RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "scale_numeric": False,
            "n_iter": 25,
            "param_distributions": {
                "model__n_estimators": [100, 150, 200, 250],
                "model__max_depth": [4, 5, 6, 7, 8, 10],
                "model__min_samples_leaf": [2, 4, 8, 12, 16],
                "model__min_samples_split": [5, 10, 15, 20],
                "model__max_features": ["sqrt", "log2", 0.4, 0.6],
                "model__class_weight": ["balanced", "balanced_subsample"],
            },
        },
        "XGBoost": {
            "estimator": XGBClassifier(
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=0,
            ),
            "scale_numeric": False,
            "n_iter": 25,
            "param_distributions": {
                "model__n_estimators": [100, 150, 200],
                "model__max_depth": [2, 3, 4],
                "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
                "model__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
                "model__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
                "model__reg_alpha": [0.01, 0.1, 0.5, 1.0, 5.0],
                "model__reg_lambda": [1.0, 3.0, 5.0, 10.0, 20.0],
                "model__scale_pos_weight": [2.3, 2.5, 2.77, 3.0],
            },
        },
    }


def evaluate_tuned_pipeline(
    pipeline: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: List[Tuple[List[int], List[int]]],
) -> Dict[str, Any]:
    """Runs rigorous 5-fold CV on the best tuned pipeline to measure all metrics and train/val gap."""
    fold_metrics = []
    train_f1s = []

    for tr_idx, val_idx in cv_splits:
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        # Clone & fit pipeline on this fold's training portion
        fold_pipe = joblib.load(joblib.dump(pipeline, None)[0]) if False else pipeline
        from sklearn.base import clone
        fold_pipe = clone(pipeline)
        fold_pipe.fit(X_tr, y_tr)

        # Train evaluation (for generalization gap)
        tr_pred = fold_pipe.predict(X_tr)
        tr_f1 = f1_score(y_tr, tr_pred, zero_division=0)
        train_f1s.append(tr_f1)

        # Val evaluation
        val_pred = fold_pipe.predict(X_val)
        val_proba = fold_pipe.predict_proba(X_val)[:, 1] if hasattr(fold_pipe, "predict_proba") else None

        fold_metrics.append({
            "f1": float(f1_score(y_val, val_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_val, val_proba)) if val_proba is not None else 0.5,
            "recall": float(recall_score(y_val, val_pred, zero_division=0)),
            "precision": float(precision_score(y_val, val_pred, zero_division=0)),
            "accuracy": float(accuracy_score(y_val, val_pred)),
        })

    mean_val_f1 = float(np.mean([m["f1"] for m in fold_metrics]))
    std_val_f1 = float(np.std([m["f1"] for m in fold_metrics]))
    mean_tr_f1 = float(np.mean(train_f1s))
    gap = round(mean_tr_f1 - mean_val_f1, 4)

    mean_auc = float(np.mean([m["roc_auc"] for m in fold_metrics]))
    mean_rec = float(np.mean([m["recall"] for m in fold_metrics]))
    mean_prec = float(np.mean([m["precision"] for m in fold_metrics]))
    mean_acc = float(np.mean([m["accuracy"] for m in fold_metrics]))

    # Check gates
    f1_gate_pass = bool(mean_val_f1 >= REGISTRY_GATES["min_f1"])
    auc_gate_pass = bool(mean_auc >= REGISTRY_GATES["min_roc_auc"])
    rec_gate_pass = bool(mean_rec >= REGISTRY_GATES["min_recall"])
    gap_gate_pass = bool(gap <= REGISTRY_GATES["max_generalization_gap"])
    all_gates_pass = bool(f1_gate_pass and auc_gate_pass and rec_gate_pass and gap_gate_pass)

    return {
        "mean_val_f1": round(mean_val_f1, 4),
        "std_val_f1": round(std_val_f1, 4),
        "mean_train_f1": round(mean_tr_f1, 4),
        "generalization_gap": gap,
        "mean_roc_auc": round(mean_auc, 4),
        "mean_recall": round(mean_rec, 4),
        "mean_precision": round(mean_prec, 4),
        "mean_accuracy": round(mean_acc, 4),
        "gate_checks": {
            "f1_pass": f1_gate_pass,
            "auc_pass": auc_gate_pass,
            "recall_pass": rec_gate_pass,
            "gap_pass": gap_gate_pass,
            "all_gates_pass": all_gates_pass,
        },
        "fold_metrics": fold_metrics,
    }


def plot_tuning_comparison(
    baseline_summary: Dict[str, Any],
    tuned_results: Dict[str, Any],
    out_path: Path,
) -> None:
    """Renders before-vs-after comparison plots for F1 score and Generalization Gap."""
    models = ["LogisticRegression", "RandomForest", "XGBoost"]
    x = np.arange(len(models))
    width = 0.35

    base_f1 = [baseline_summary.get(m, {}).get("f1_mean", 0.0) for m in models]
    tuned_f1 = [tuned_results[m]["metrics"]["mean_val_f1"] for m in models]

    # Baseline gaps from cv_analysis_report if available
    base_gaps = [0.0067, 0.4494, 0.3788]  # from Phase 10 / 11
    tuned_gaps = [tuned_results[m]["metrics"]["generalization_gap"] for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: F1 Comparison
    ax1 = axes[0]
    b1 = ax1.bar(x - width / 2, base_f1, width, label="Baseline", color="#94A3B8", alpha=0.85)
    b2 = ax1.bar(x + width / 2, tuned_f1, width, label="Tuned (Phase 12)", color="#3B82F6", alpha=0.9)
    ax1.axhline(0.60, color="red", ls="--", label="Registry Gate (0.60)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontweight="medium")
    ax1.set_ylabel("Validation F1 Score", fontsize=10)
    ax1.set_title("Validation F1: Baseline vs. Tuned", fontweight="bold", fontsize=11)
    ax1.set_ylim(0, 0.75)
    ax1.grid(alpha=0.3, axis="y")
    ax1.legend(loc="lower right")

    for bar in b1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=8.5)
    for bar in b2:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Panel 2: Generalization Gap Comparison
    ax2 = axes[1]
    g1 = ax2.bar(x - width / 2, base_gaps, width, label="Baseline Gap", color="#EF4444", alpha=0.85)
    g2 = ax2.bar(x + width / 2, tuned_gaps, width, label="Tuned Gap", color="#10B981", alpha=0.9)
    ax2.axhline(0.08, color="red", ls="--", label="Max Gap Gate (0.08)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontweight="medium")
    ax2.set_ylabel("Generalization Gap (Train F1 - Val F1)", fontsize=10)
    ax2.set_title("Generalization Gap: Overfitting Eradication", fontweight="bold", fontsize=11)
    ax2.set_ylim(-0.05, 0.52)
    ax2.grid(alpha=0.3, axis="y")
    ax2.legend(loc="upper right")

    for bar in g1:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:+.3f}", ha="center", va="bottom", fontsize=8.5)
    for bar in g2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:+.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.suptitle("Phase 12 — Hyperparameter Optimization: Baseline vs. Regularized Tuned Models",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved: %s", out_path)


def run_tuning() -> Dict[str, Any]:
    """Orchestrates Phase 12 Hyperparameter Optimization across all candidate models."""
    log.info("=" * 60)
    log.info("Phase 12 — Hyperparameter Optimization (HPO)")
    log.info("=" * 60)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X, y, cv_splits = _load_data_and_folds()
    search_spaces = get_search_spaces()

    # Load baseline summary for before-after comparison
    baseline_path = REPORTS_DIR / "baseline_summary.json"
    baseline_summary: Dict[str, Any] = {}
    if baseline_path.exists():
        with open(baseline_path, "r", encoding="utf-8") as f:
            baseline_summary = json.load(f)

    tuned_results: Dict[str, Any] = {}

    for model_name, cfg in search_spaces.items():
        log.info("\n>>> Optimizing %s (n_iter=%d)...", model_name, cfg["n_iter"])
        t0 = time.perf_counter()

        # Build initial full pipeline with this model's base estimator
        base_pipeline = build_full_pipeline(
            estimator=cfg["estimator"],
            scale_numeric=cfg["scale_numeric"],
        )

        # Run RandomizedSearchCV across pre-partitioned folds
        search = RandomizedSearchCV(
            estimator=base_pipeline,
            param_distributions=cfg["param_distributions"],
            n_iter=cfg["n_iter"],
            scoring="f1",
            cv=cv_splits,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            refit=True,
            verbose=0,
        )
        search.fit(X, y)
        elapsed = time.perf_counter() - t0

        best_pipeline = search.best_estimator_
        best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
        log.info("  Optimized in %.1fs | Best CV F1: %.4f", elapsed, search.best_score_)
        log.info("  Best params: %s", best_params)

        # Evaluate best pipeline rigorously across all folds
        metrics = evaluate_tuned_pipeline(best_pipeline, X, y, cv_splits)
        log.info(
            "  Val F1: %.4f ± %.4f | Train F1: %.4f | Gap: %+.4f (Gate <=0.08: %s)",
            metrics["mean_val_f1"],
            metrics["std_val_f1"],
            metrics["mean_train_f1"],
            metrics["generalization_gap"],
            "PASS" if metrics["gate_checks"]["gap_pass"] else "FAIL",
        )

        # Refit on 100% of train.csv (strictly train data, 5,634 rows) and save artifact
        best_pipeline.fit(X, y)
        model_filename = f"tuned_{model_name.lower()}.joblib"
        save_path = MODELS_DIR / model_filename
        joblib.dump(best_pipeline, save_path)
        log.info("  Saved tuned model pipeline -> %s", save_path)

        tuned_results[model_name] = {
            "search_time_seconds": round(elapsed, 1),
            "best_params": best_params,
            "metrics": metrics,
            "model_path": str(save_path),
        }

    # Generate comparison plot
    plot_path = REPORTS_DIR / "tuning_comparison.png"
    plot_tuning_comparison(baseline_summary, tuned_results, plot_path)

    # Compile structured JSON tuning report
    report: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": 12,
        "sample_count": len(X),
        "scoring": "f1",
        "cv_folds": len(cv_splits),
        "registry_gates": REGISTRY_GATES,
        "tuned_models": tuned_results,
    }

    report_path = REPORTS_DIR / "tuning_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log.info("\nSaved full tuning report -> %s", report_path)

    return report


if __name__ == "__main__":
    t_start = time.perf_counter()
    report = run_tuning()
    total_elapsed = time.perf_counter() - t_start

    print("\n" + "=" * 75)
    print(f"PHASE 12 — HYPERPARAMETER OPTIMIZATION COMPLETE  |  Total Time: {total_elapsed:.1f}s")
    print("=" * 75)
    for model, res in report["tuned_models"].items():
        m = res["metrics"]
        g = m["gate_checks"]
        status = "PASSED ALL GATES" if g["all_gates_pass"] else "GATES NOT MET"
        print(f"  {model:<20} | Val F1: {m['mean_val_f1']:.4f} | Gap: {m['generalization_gap']:+.4f} | AUC: {m['mean_roc_auc']:.4f} | Recall: {m['mean_recall']:.4f} | {status}")
    print("=" * 75)
