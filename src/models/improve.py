"""
Phase 13 — Iterative Improvement
================================
Implements advanced modeling iterations to push beyond single-model tuning:
  1. Ensemble Modeling:
     - Soft-Voting Classifier (Weighted probability blend of Tuned LR, RF, XGBoost)
     - Stacking Classifier (Base models -> LogisticRegression meta-learner)
  2. Decision Threshold Optimization:
     - Out-of-fold search across thresholds T in [0.25, 0.75]
     - Discovers F1-maximizing threshold T* for each model and ensemble
  3. 5-Fold Cross-Validation Audit:
     - Measures F1, ROC-AUC, Recall, Precision, and Generalization Gap
  4. Artifact Serialization:
     - models/ensemble_voting.joblib
     - models/ensemble_stacking.joblib
     - reports/improvement_report.json
     - reports/ensemble_comparison.png

Leakage Rule (R3):
  Strictly evaluated on data/processed/train.csv and cv_folds.json.
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
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from src.data.preprocessing import build_full_pipeline
from src.utils.config import PARAMS
from src.utils.logging import get_logger

log = get_logger("improve")

RANDOM_STATE: int = PARAMS["random_state"]
TARGET: str = PARAMS["dataset"]["target_column"]
GAP_THRESHOLD: float = PARAMS["diagnosis"]["gap_threshold"]
REGISTRY_GATES: Dict[str, float] = PARAMS["registry"]
REPORTS_DIR = Path("reports")
MODELS_DIR = Path("models")


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


def get_tuned_pipelines() -> Dict[str, Any]:
    """Instantiates the best tuned pipelines from Phase 12 HPO."""
    lr_pipe = build_full_pipeline(
        estimator=LogisticRegression(
            C=0.005,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        scale_numeric=True,
    )
    rf_pipe = build_full_pipeline(
        estimator=RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=2,
            min_samples_split=20,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        scale_numeric=False,
    )
    xgb_pipe = build_full_pipeline(
        estimator=XGBClassifier(
            n_estimators=200,
            max_depth=2,
            learning_rate=0.08,
            scale_pos_weight=2.5,
            subsample=0.7,
            colsample_bytree=1.0,
            reg_alpha=1.0,
            reg_lambda=20.0,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        ),
        scale_numeric=False,
    )
    return {
        "LogisticRegression": lr_pipe,
        "RandomForest": rf_pipe,
        "XGBoost": xgb_pipe,
    }


def evaluate_model_cv(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: List[Tuple[List[int], List[int]]],
) -> Dict[str, Any]:
    """
    Evaluates a model/ensemble across the 5 CV folds.
    Returns default threshold (0.50) metrics and optimal threshold metrics.
    """
    n_samples = len(X)
    oof_probs = np.zeros(n_samples)
    oof_train_f1s = []
    fold_metrics = []

    for tr_idx, val_idx in cv_splits:
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        m_clone = clone(model)
        m_clone.fit(X_tr, y_tr)

        # Train evaluation
        tr_pred = m_clone.predict(X_tr)
        oof_train_f1s.append(f1_score(y_tr, tr_pred, zero_division=0))

        # Val evaluation
        val_proba = m_clone.predict_proba(X_val)[:, 1]
        oof_probs[val_idx] = val_proba

        val_pred = (val_proba >= 0.50).astype(int)
        fold_metrics.append({
            "f1": float(f1_score(y_val, val_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_val, val_proba)),
            "recall": float(recall_score(y_val, val_pred, zero_division=0)),
            "precision": float(precision_score(y_val, val_pred, zero_division=0)),
            "accuracy": float(accuracy_score(y_val, val_pred)),
        })

    # Default threshold (0.50) summary
    mean_val_f1 = float(np.mean([m["f1"] for m in fold_metrics]))
    std_val_f1 = float(np.std([m["f1"] for m in fold_metrics]))
    mean_train_f1 = float(np.mean(oof_train_f1s))
    gap = round(mean_train_f1 - mean_val_f1, 4)
    mean_auc = float(np.mean([m["roc_auc"] for m in fold_metrics]))
    mean_rec = float(np.mean([m["recall"] for m in fold_metrics]))
    mean_prec = float(np.mean([m["precision"] for m in fold_metrics]))
    mean_acc = float(np.mean([m["accuracy"] for m in fold_metrics]))

    # Threshold Optimization across OOF predictions
    thresholds = np.linspace(0.20, 0.80, 61)
    best_t = 0.50
    best_f1 = 0.0
    best_rec = 0.0
    best_prec = 0.0

    for t in thresholds:
        preds_t = (oof_probs >= t).astype(int)
        f = f1_score(y, preds_t, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_t = float(t)
            best_rec = float(recall_score(y, preds_t, zero_division=0))
            best_prec = float(precision_score(y, preds_t, zero_division=0))

    return {
        "default_threshold": 0.50,
        "mean_val_f1": round(mean_val_f1, 4),
        "std_val_f1": round(std_val_f1, 4),
        "mean_train_f1": round(mean_train_f1, 4),
        "generalization_gap": gap,
        "mean_roc_auc": round(mean_auc, 4),
        "mean_recall": round(mean_rec, 4),
        "mean_precision": round(mean_prec, 4),
        "mean_accuracy": round(mean_acc, 4),
        "optimal_threshold": round(best_t, 3),
        "f1_at_optimal_threshold": round(best_f1, 4),
        "recall_at_optimal_threshold": round(best_rec, 4),
        "precision_at_optimal_threshold": round(best_prec, 4),
        "f1_gain_from_threshold": round(best_f1 - mean_val_f1, 4),
        "gate_checks": {
            "f1_pass": bool(mean_val_f1 >= REGISTRY_GATES["min_f1"]),
            "auc_pass": bool(mean_auc >= REGISTRY_GATES["min_roc_auc"]),
            "recall_pass": bool(mean_rec >= REGISTRY_GATES["min_recall"]),
            "gap_pass": bool(gap <= REGISTRY_GATES["max_generalization_gap"]),
            "all_gates_pass": bool(
                mean_val_f1 >= REGISTRY_GATES["min_f1"]
                and mean_auc >= REGISTRY_GATES["min_roc_auc"]
                and mean_rec >= REGISTRY_GATES["min_recall"]
                and gap <= REGISTRY_GATES["max_generalization_gap"]
            ),
        },
    }


def plot_improvement_comparison(results: Dict[str, Any], out_path: Path) -> None:
    """Plots comparative performance: default 0.50 vs. optimal threshold F1 and ROC-AUC."""
    models = list(results.keys())
    x = np.arange(len(models))
    width = 0.35

    f1_def = [results[m]["mean_val_f1"] for m in models]
    f1_opt = [results[m]["f1_at_optimal_threshold"] for m in models]
    opt_t = [results[m]["optimal_threshold"] for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Panel 1: Default vs Optimal Threshold F1
    ax1 = axes[0]
    b1 = ax1.bar(x - width / 2, f1_def, width, label="F1 at Threshold 0.50", color="#94A3B8", alpha=0.85)
    b2 = ax1.bar(x + width / 2, f1_opt, width, label="F1 at Optimal Threshold T*", color="#3B82F6", alpha=0.9)
    ax1.axhline(0.60, color="red", ls="--", label="Registry Gate (0.60)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace("Classifier", "") for m in models], rotation=15, ha="right", fontsize=9.5)
    ax1.set_ylabel("Validation F1 Score", fontsize=10)
    ax1.set_title("Validation F1: Default (0.50) vs. Optimal Threshold T*", fontweight="bold", fontsize=11)
    ax1.set_ylim(0, 0.75)
    ax1.grid(alpha=0.3, axis="y")
    ax1.legend(loc="lower right")

    for bar, t in zip(b2, opt_t):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}\n(T*={t:.2f})", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for bar in b1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    # Panel 2: ROC-AUC and Generalization Gap
    ax2 = axes[1]
    aucs = [results[m]["mean_roc_auc"] for m in models]
    gaps = [results[m]["generalization_gap"] for m in models]
    b_auc = ax2.bar(x - width / 2, aucs, width, label="Validation ROC-AUC", color="#10B981", alpha=0.9)
    b_gap = ax2.bar(x + width / 2, gaps, width, label="Generalization Gap", color="#F59E0B", alpha=0.9)
    ax2.axhline(0.75, color="green", ls=":", label="Min AUC Gate (0.75)")
    ax2.axhline(0.08, color="red", ls="--", label="Max Gap Gate (0.08)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.replace("Classifier", "") for m in models], rotation=15, ha="right", fontsize=9.5)
    ax2.set_ylabel("Metric Value", fontsize=10)
    ax2.set_title("ROC-AUC & Generalization Gap Compliance", fontweight="bold", fontsize=11)
    ax2.set_ylim(-0.02, 0.95)
    ax2.grid(alpha=0.3, axis="y")
    ax2.legend(loc="upper right")

    for bar in b_auc:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for bar in b_gap:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:+.3f}", ha="center", va="bottom", fontsize=8)

    plt.suptitle("Phase 13 — Iterative Improvement: Ensembles & Threshold Optimization",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved: %s", out_path)


def run_improvement() -> Dict[str, Any]:
    """Executes Phase 13 Iterative Improvement."""
    log.info("=" * 60)
    log.info("Phase 13 — Iterative Improvement")
    log.info("=" * 60)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X, y, cv_splits = _load_data_and_folds()
    base_pipes = get_tuned_pipelines()

    # Build Ensembles
    # 1. Soft Voting Classifier
    voting_clf = VotingClassifier(
        estimators=[
            ("lr", base_pipes["LogisticRegression"]),
            ("rf", base_pipes["RandomForest"]),
            ("xgb", base_pipes["XGBoost"]),
        ],
        voting="soft",
        weights=[1.0, 1.0, 1.2],  # Slightly higher weight for leading XGBoost
    )

    # 2. Stacking Classifier
    stacking_clf = StackingClassifier(
        estimators=[
            ("lr", base_pipes["LogisticRegression"]),
            ("rf", base_pipes["RandomForest"]),
            ("xgb", base_pipes["XGBoost"]),
        ],
        final_estimator=LogisticRegression(C=0.1, random_state=RANDOM_STATE),
        cv=5,
        n_jobs=-1,
        passthrough=False,
    )


    all_candidates = {
        "LogisticRegression": base_pipes["LogisticRegression"],
        "RandomForest":       base_pipes["RandomForest"],
        "XGBoost":            base_pipes["XGBoost"],
        "VotingClassifier":   voting_clf,
        "StackingClassifier": stacking_clf,
    }

    results: Dict[str, Any] = {}

    for name, model in all_candidates.items():
        log.info("\n>>> Evaluating %s across 5-fold CV...", name)
        t0 = time.perf_counter()
        metrics = evaluate_model_cv(model, X, y, cv_splits)
        elapsed = time.perf_counter() - t0

        log.info("  Default F1 (0.50): %.4f ± %.4f | Gap: %+.4f | AUC: %.4f",
                 metrics["mean_val_f1"], metrics["std_val_f1"],
                 metrics["generalization_gap"], metrics["mean_roc_auc"])
        log.info("  Optimal T*: %.3f -> F1: %.4f (Gain: %+.4f) | Recall: %.4f",
                 metrics["optimal_threshold"], metrics["f1_at_optimal_threshold"],
                 metrics["f1_gain_from_threshold"], metrics["recall_at_optimal_threshold"])

        results[name] = metrics

    # Refit and serialize Ensembles on 100% of train.csv
    log.info("\nRefitting ensembles on full train.csv (5,634 rows)...")
    voting_clf.fit(X, y)
    voting_path = MODELS_DIR / "ensemble_voting.joblib"
    joblib.dump(voting_clf, voting_path)
    log.info("Saved -> %s", voting_path)

    stacking_clf.fit(X, y)
    stacking_path = MODELS_DIR / "ensemble_stacking.joblib"
    joblib.dump(stacking_clf, stacking_path)
    log.info("Saved -> %s", stacking_path)

    # Visualization
    plot_path = REPORTS_DIR / "ensemble_comparison.png"
    plot_improvement_comparison(results, plot_path)

    # Find the top overall candidate by optimal F1
    best_candidate_name = max(results.keys(), key=lambda k: results[k]["f1_at_optimal_threshold"])
    best_candidate_data = results[best_candidate_name]

    # Structured JSON report
    report: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": 13,
        "train_rows": len(X),
        "cv_folds": len(cv_splits),
        "registry_gates": REGISTRY_GATES,
        "models": results,
        "champion_candidate": {
            "model_name": best_candidate_name,
            "optimal_threshold": best_candidate_data["optimal_threshold"],
            "f1_at_optimal": best_candidate_data["f1_at_optimal_threshold"],
            "roc_auc": best_candidate_data["mean_roc_auc"],
            "recall": best_candidate_data["recall_at_optimal_threshold"],
            "gap": best_candidate_data["generalization_gap"],
        },
    }

    report_path = REPORTS_DIR / "improvement_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log.info("Saved report -> %s", report_path)

    return report


if __name__ == "__main__":
    t_start = time.perf_counter()
    report = run_improvement()
    total_elapsed = time.perf_counter() - t_start

    print("\n" + "=" * 80)
    print(f"PHASE 13 — ITERATIVE IMPROVEMENT COMPLETE  |  Total Time: {total_elapsed:.1f}s")
    print("=" * 80)
    for model, m in report["models"].items():
        print(f"  {model:<20} | F1@0.50: {m['mean_val_f1']:.4f} | Opt T*: {m['optimal_threshold']:.2f} -> F1: {m['f1_at_optimal_threshold']:.4f} | AUC: {m['mean_roc_auc']:.4f} | Gap: {m['generalization_gap']:+.4f}")
    champ = report["champion_candidate"]
    print("-" * 80)
    print(f"Top Candidate: {champ['model_name']} with F1={champ['f1_at_optimal']:.4f} at T*={champ['optimal_threshold']:.2f}")
    print("=" * 80)
