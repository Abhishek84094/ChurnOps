"""
Phase 9 - Baseline Model Training
==================================
Trains four baseline classifiers using full end-to-end Scikit-Learn Pipelines
(DomainFeatureEngineer -> ColumnTransformer -> Estimator).

Models:
  1. DummyClassifier       - stratified random baseline
  2. LogisticRegression    - scaled pipeline, class_weight='balanced'
  3. RandomForestClassifier- unscaled pipeline, class_weight='balanced'
  4. XGBClassifier         - unscaled pipeline, scale_pos_weight per fold

Imbalance Handling (Rule R3 / params.yaml imbalance.strategy = class_weight):
  LR / RF : class_weight='balanced'
  XGB     : scale_pos_weight = neg/pos (recomputed each fold from training labels)
  Dummy   : not applicable

Leakage (Rule R3):
  Each CV fold fits the ENTIRE pipeline on the training split of that fold only.
  test.csv is NEVER accessed here.

Outputs:
  reports/baseline_cv_results.json
  reports/baseline_summary.json
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

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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

log = get_logger("train_baseline")

RANDOM_STATE: int = PARAMS["random_state"]
TARGET: str = PARAMS["dataset"]["target_column"]


def _compute_scale_pos_weight(y: pd.Series) -> float:
    """XGBoost imbalance correction: ratio of negatives to positives."""
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    return float(neg / pos) if pos > 0 else 1.0


def build_model_pipelines(y_train: pd.Series) -> Dict[str, Any]:
    """Returns a dict of {model_name: Pipeline} for each baseline model."""
    spw = _compute_scale_pos_weight(y_train)
    log.info("Initial XGBoost scale_pos_weight: %.4f", spw)

    models: Dict[str, Any] = {
        "DummyClassifier": build_full_pipeline(
            estimator=DummyClassifier(strategy="stratified", random_state=RANDOM_STATE),
            scale_numeric=True,
        ),
        "LogisticRegression": build_full_pipeline(
            estimator=LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                solver="lbfgs",
                random_state=RANDOM_STATE,
            ),
            scale_numeric=True,
        ),
        "RandomForest": build_full_pipeline(
            estimator=RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            scale_numeric=False,
        ),
        "XGBoost": build_full_pipeline(
            estimator=XGBClassifier(
                n_estimators=200,
                scale_pos_weight=spw,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=0,
            ),
            scale_numeric=False,
        ),
    }
    return models


def evaluate_fold(
    pipeline: Any,
    X_train_fold: pd.DataFrame,
    y_train_fold: pd.Series,
    X_val_fold: pd.DataFrame,
    y_val_fold: pd.Series,
) -> Dict[str, float]:
    """Fits pipeline on training fold, evaluates on validation fold."""
    pipeline.fit(X_train_fold, y_train_fold)
    y_pred = pipeline.predict(X_val_fold)
    try:
        y_prob = pipeline.predict_proba(X_val_fold)[:, 1]
    except AttributeError:
        y_prob = pipeline.decision_function(X_val_fold)

    return {
        "accuracy":  round(accuracy_score(y_val_fold, y_pred), 4),
        "precision": round(precision_score(y_val_fold, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_val_fold, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_val_fold, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_val_fold, y_prob), 4),
    }


def cross_validate_models(
    df_train: pd.DataFrame,
    cv_folds: Dict[str, Dict[str, List[int]]],
) -> Tuple[Dict[str, List[Dict[str, float]]], Dict[str, Dict[str, str]]]:
    """
    Runs stratified k-fold CV for all four baseline models.
    cv_folds format: {"fold_1": {"train_indices": [...], "val_indices": [...]}, ...}
    Each fold rebuilds and fits an independent pipeline - no leakage.
    """
    X = df_train.drop(columns=[TARGET])
    y = df_train[TARGET]

    pipelines = build_model_pipelines(y)
    fold_names = sorted(cv_folds.keys())  # fold_1, fold_2, ...
    fold_results: Dict[str, List[Dict[str, float]]] = {name: [] for name in pipelines}

    for fold_idx, fold_name in enumerate(fold_names):
        fold = cv_folds[fold_name]
        X_tr  = X.iloc[fold["train_indices"]].reset_index(drop=True)
        y_tr  = y.iloc[fold["train_indices"]].reset_index(drop=True)
        X_val = X.iloc[fold["val_indices"]].reset_index(drop=True)
        y_val = y.iloc[fold["val_indices"]].reset_index(drop=True)

        log.info(
            "Fold %d/%d | train=%d val=%d pos_rate_train=%.3f pos_rate_val=%.3f",
            fold_idx + 1, len(fold_names), len(y_tr), len(y_val), y_tr.mean(), y_val.mean(),
        )

        for name, pipe in pipelines.items():
            if name == "XGBoost":
                spw_fold = _compute_scale_pos_weight(y_tr)
                pipe.named_steps["model"].set_params(scale_pos_weight=spw_fold)

            t0 = time.perf_counter()
            metrics = evaluate_fold(pipe, X_tr, y_tr, X_val, y_val)
            elapsed = round(time.perf_counter() - t0, 2)
            metrics["train_time_s"] = elapsed

            fold_results[name].append(metrics)
            log.info(
                "  %-22s | F1=%.4f  AUC=%.4f  Recall=%.4f  [%.1fs]",
                name, metrics["f1"], metrics["roc_auc"], metrics["recall"], elapsed,
            )

    summary_results: Dict[str, Dict[str, str]] = {}
    for name, folds in fold_results.items():
        metric_keys = [k for k in folds[0] if k != "train_time_s"]
        summary: Dict[str, str] = {}
        for k in metric_keys:
            vals = [f[k] for f in folds]
            summary[k] = f"{np.mean(vals):.4f} +/- {np.std(vals):.4f}"
        summary["mean_train_time_s"] = f"{np.mean([f['train_time_s'] for f in folds]):.2f}"
        summary_results[name] = summary

    return fold_results, summary_results


def run_baseline_training() -> Dict[str, Any]:
    """Main entry point for Phase 9."""
    log.info("=" * 60)
    log.info("Phase 9 - Baseline Model Training")
    log.info("=" * 60)

    train_path = Path("data/processed/train.csv")
    folds_path = Path("data/processed/cv_folds.json")

    if not train_path.exists():
        raise FileNotFoundError(f"train.csv not found: {train_path}")
    if not folds_path.exists():
        raise FileNotFoundError(f"cv_folds.json not found: {folds_path}")

    df_train = pd.read_csv(train_path)
    with open(folds_path, "r", encoding="utf-8") as f:
        cv_folds = json.load(f)

    log.info("Loaded train.csv: %d rows, %d cols", *df_train.shape)
    log.info("Loaded %d CV folds", len(cv_folds))

    pos_count = int((df_train[TARGET] == 1).sum())
    neg_count = int((df_train[TARGET] == 0).sum())
    class_ratio = round(neg_count / pos_count, 4)
    log.info("Class distribution: pos=%d neg=%d ratio=%.4f", pos_count, neg_count, class_ratio)

    fold_results, summary_results = cross_validate_models(df_train, cv_folds)

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    cv_report = {
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "phase":         9,
        "n_folds":       len(cv_folds),
        "train_rows":    len(df_train),
        "class_balance": {
            "positive": pos_count,
            "negative": neg_count,
            "neg_pos_ratio": class_ratio,
        },
        "imbalance_strategy": {
            "LR_RF":  "class_weight='balanced'",
            "XGBoost": f"scale_pos_weight={class_ratio}",
            "Dummy":  "N/A",
        },
        "fold_results": fold_results,
    }

    cv_path = reports_dir / "baseline_cv_results.json"
    with open(cv_path, "w", encoding="utf-8") as f:
        json.dump(cv_report, f, indent=2)
    log.info("Saved CV results -> %s", cv_path)

    summary_path = reports_dir / "baseline_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": datetime.now(timezone.utc).isoformat(),
                   "phase": 9, "summary": summary_results}, f, indent=2)
    log.info("Saved summary -> %s", summary_path)

    return summary_results


if __name__ == "__main__":
    summary = run_baseline_training()
    print("\n" + "=" * 70)
    print("PHASE 9 - BASELINE MODELS  |  5-Fold CV Summary")
    print("=" * 70)
    print(f"{'Model':<24} {'F1':>12} {'ROC-AUC':>14} {'Recall':>12} {'Precision':>12}")
    print("-" * 70)
    for model_name, metrics in summary.items():
        print(
            f"{model_name:<24} "
            f"{metrics.get('f1', 'N/A'):>12} "
            f"{metrics.get('roc_auc', 'N/A'):>14} "
            f"{metrics.get('recall', 'N/A'):>12} "
            f"{metrics.get('precision', 'N/A'):>12}"
        )
    print("=" * 70)
