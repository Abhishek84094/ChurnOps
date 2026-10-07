"""
Phase 24 — Prediction Quality Monitor
=======================================
Tracks model prediction quality over time using:
  - Calibration check (predicted probability vs observed churn rate)
  - Score distribution shift
  - Business metric thresholds (churn rate alerts)

For labeled data, compute actual vs predicted metrics.
For unlabeled data (production), track score distribution statistics.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score, f1_score, recall_score

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.config import PARAMS
from src.utils.logging import get_logger

log = get_logger("quality_monitor")

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
DATA_DIR = Path("data")

DECISION_THRESHOLD = 0.48
TARGET_COL = PARAMS["dataset"]["target_column"]

ALERT_THRESHOLDS = {
    "max_churn_rate": 0.40,       # Alert if predicted churn > 40% of batch
    "min_roc_auc": 0.75,          # Alert if AUC drops below this
    "max_brier_score": 0.20,      # Alert if Brier score exceeds this (poor calibration)
    "max_mean_proba_shift": 0.10, # Alert if mean predicted probability shifts > 10%
}

# Baseline reference from training (Phase 14)
REFERENCE_STATS = {
    "mean_churn_proba": 0.2654,   # Training set churn rate
    "brier_score": 0.1644,        # From Phase 14 evaluation
    "roc_auc": 0.8491,
}


def compute_score_stats(probas: np.ndarray) -> Dict[str, Any]:
    """Compute distribution statistics of predicted probabilities."""
    return {
        "count": len(probas),
        "mean": round(float(probas.mean()), 4),
        "std": round(float(probas.std()), 4),
        "min": round(float(probas.min()), 4),
        "max": round(float(probas.max()), 4),
        "p25": round(float(np.percentile(probas, 25)), 4),
        "p50": round(float(np.percentile(probas, 50)), 4),
        "p75": round(float(np.percentile(probas, 75)), 4),
        "high_risk_rate": round(float((probas >= DECISION_THRESHOLD).mean()), 4),
    }


def evaluate_labeled_quality(
    y_true: np.ndarray, probas: np.ndarray
) -> Dict[str, Any]:
    """Compute quality metrics when ground truth labels are available."""
    y_pred = (probas >= DECISION_THRESHOLD).astype(int)

    roc_auc = float(roc_auc_score(y_true, probas)) if len(np.unique(y_true)) > 1 else 0.0
    brier = float(brier_score_loss(y_true, probas))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))

    # Calibration curve (5 bins)
    try:
        fraction_pos, mean_pred = calibration_curve(y_true, probas, n_bins=5)
        calibration_data = {
            "fraction_positives": [round(float(x), 4) for x in fraction_pos],
            "mean_predicted_values": [round(float(x), 4) for x in mean_pred],
        }
    except Exception:
        calibration_data = {}

    return {
        "roc_auc": round(roc_auc, 4),
        "brier_score": round(brier, 4),
        "f1_score": round(f1, 4),
        "recall": round(recall, 4),
        "calibration": calibration_data,
    }


def generate_alerts(
    score_stats: Dict[str, Any],
    labeled_metrics: Optional[Dict[str, Any]] = None,
) -> list:
    """Generate actionable alerts based on quality thresholds."""
    alerts = []

    if score_stats["high_risk_rate"] > ALERT_THRESHOLDS["max_churn_rate"]:
        alerts.append({
            "level": "WARNING",
            "type": "HIGH_CHURN_RATE",
            "message": (
                f"Predicted churn rate {score_stats['high_risk_rate']:.1%} exceeds threshold "
                f"{ALERT_THRESHOLDS['max_churn_rate']:.1%}. Investigate business conditions."
            ),
        })

    mean_shift = abs(score_stats["mean"] - REFERENCE_STATS["mean_churn_proba"])
    if mean_shift > ALERT_THRESHOLDS["max_mean_proba_shift"]:
        alerts.append({
            "level": "WARNING",
            "type": "SCORE_DISTRIBUTION_SHIFT",
            "message": (
                f"Mean predicted probability shifted by {mean_shift:.3f} "
                f"(ref: {REFERENCE_STATS['mean_churn_proba']:.3f}, current: {score_stats['mean']:.3f}). "
                "Consider retraining."
            ),
        })

    if labeled_metrics:
        if labeled_metrics["roc_auc"] < ALERT_THRESHOLDS["min_roc_auc"]:
            alerts.append({
                "level": "CRITICAL",
                "type": "ROC_AUC_DEGRADATION",
                "message": (
                    f"ROC-AUC {labeled_metrics['roc_auc']:.4f} < threshold {ALERT_THRESHOLDS['min_roc_auc']}. "
                    "Model performance has degraded — retraining required."
                ),
            })
        if labeled_metrics["brier_score"] > ALERT_THRESHOLDS["max_brier_score"]:
            alerts.append({
                "level": "WARNING",
                "type": "POOR_CALIBRATION",
                "message": (
                    f"Brier score {labeled_metrics['brier_score']:.4f} > threshold "
                    f"{ALERT_THRESHOLDS['max_brier_score']}. Predicted probabilities are poorly calibrated."
                ),
            })

    return alerts


def run_quality_check(
    data: pd.DataFrame,
    has_labels: bool = True,
) -> dict:
    """Run complete quality check on a prediction dataset."""
    pipeline = joblib.load(MODELS_DIR / "champion_model.joblib")

    target_col = TARGET_COL
    if has_labels and target_col in data.columns:
        y_true = data[target_col].values
        X = data.drop(columns=[target_col])
    else:
        y_true = None
        X = data.drop(columns=[target_col], errors="ignore")

    probas = pipeline.predict_proba(X)[:, 1]
    score_stats = compute_score_stats(probas)

    labeled_metrics = None
    if y_true is not None:
        labeled_metrics = evaluate_labeled_quality(y_true, probas)

    alerts = generate_alerts(score_stats, labeled_metrics)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": 24,
        "samples_evaluated": len(data),
        "score_statistics": score_stats,
        "performance_metrics": labeled_metrics,
        "alerts": alerts,
        "alert_count": len(alerts),
        "status": "ALERT" if alerts else "HEALTHY",
    }


def main():
    t0 = time.perf_counter()
    log.info("=" * 70)
    log.info("PHASE 24 — PREDICTION QUALITY MONITORING")
    log.info("=" * 70)

    test_path = DATA_DIR / "processed" / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found at {test_path}")

    test_data = pd.read_csv(test_path)
    report = run_quality_check(test_data, has_labels=True)
    elapsed = time.perf_counter() - t0

    out_path = REPORTS_DIR / "quality_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log.info("Quality report saved -> %s", out_path)

    print("\n" + "=" * 80)
    print(f"PHASE 24 — QUALITY MONITORING COMPLETE  |  {elapsed:.2f}s")
    print("=" * 80)
    print(f"  Samples        : {report['samples_evaluated']}")
    print(f"  Status         : {report['status']}")
    print(f"  Alerts         : {report['alert_count']}")
    if report.get("performance_metrics"):
        pm = report["performance_metrics"]
        print(f"  ROC-AUC        : {pm['roc_auc']}")
        print(f"  F1 Score       : {pm['f1_score']}")
        print(f"  Brier Score    : {pm['brier_score']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
