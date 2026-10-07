"""
Phase 24 — Data Drift Detection
=================================
Monitors incoming prediction data for drift relative to the training distribution.
Uses Population Stability Index (PSI) for numeric features and chi-squared
tests for categorical features.

Outputs:
  - reports/drift_report.json: full drift report with PSI scores and p-values
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone


class _NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy scalar types."""
    def default(self, obj):
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.config import PARAMS
from src.utils.logging import get_logger

log = get_logger("drift_monitor")

REPORTS_DIR = Path("reports")
DATA_DIR = Path("data")

PSI_THRESHOLDS = {
    "stable": 0.10,      # PSI < 0.10 → no significant drift
    "moderate": 0.25,    # 0.10 <= PSI < 0.25 → moderate drift — monitor
    "severe": float("inf"),  # PSI >= 0.25 → severe drift — retrain
}

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "Contract", "PaymentMethod", "InternetService",
    "Partner", "Dependents", "SeniorCitizen",
]


def compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Compute Population Stability Index (PSI) between two distributions."""
    eps = 1e-6

    # Create quantile-based bins from reference distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints = np.unique(breakpoints)  # deduplicate flat regions
    if len(breakpoints) < 2:
        return 0.0

    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    # Clip zeros
    expected_pct = np.clip(expected_pct, eps, None)
    actual_pct = np.clip(actual_pct, eps, None)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(round(psi, 6))


def classify_drift(psi: float) -> str:
    """Classify drift level from PSI score."""
    if psi < PSI_THRESHOLDS["stable"]:
        return "STABLE"
    elif psi < PSI_THRESHOLDS["moderate"]:
        return "MODERATE"
    else:
        return "SEVERE"


def analyze_numeric_drift(
    reference: pd.DataFrame, current: pd.DataFrame
) -> Dict[str, Any]:
    """PSI + KS test for numeric features."""
    results = {}
    for feat in NUMERIC_FEATURES:
        if feat not in reference.columns or feat not in current.columns:
            continue

        ref_vals = reference[feat].dropna().values
        cur_vals = current[feat].dropna().values

        psi = compute_psi(ref_vals, cur_vals)
        ks_stat, ks_pval = ks_2samp(ref_vals, cur_vals)

        results[feat] = {
            "psi": psi,
            "drift_level": classify_drift(psi),
            "ks_statistic": round(float(ks_stat), 6),
            "ks_pvalue": round(float(ks_pval), 6),
            "ks_drift_detected": bool(ks_pval < 0.05),
            "reference_mean": round(float(ref_vals.mean()), 4),
            "reference_std": round(float(ref_vals.std()), 4),
            "current_mean": round(float(cur_vals.mean()), 4),
            "current_std": round(float(cur_vals.std()), 4),
        }
    return results


def analyze_categorical_drift(
    reference: pd.DataFrame, current: pd.DataFrame
) -> Dict[str, Any]:
    """Chi-squared test for categorical features."""
    results = {}
    for feat in CATEGORICAL_FEATURES:
        if feat not in reference.columns or feat not in current.columns:
            continue

        ref_counts = reference[feat].value_counts()
        cur_counts = current[feat].value_counts()

        # Align categories
        all_cats = set(ref_counts.index) | set(cur_counts.index)
        ref_aligned = np.array([ref_counts.get(c, 0) for c in all_cats], dtype=float)
        cur_aligned = np.array([cur_counts.get(c, 0) for c in all_cats], dtype=float)

        # Chi-squared test
        try:
            contingency = np.stack([ref_aligned, cur_aligned])
            chi2, pval, dof, _ = chi2_contingency(contingency)
            drift_detected = pval < 0.05
        except Exception:
            chi2, pval, drift_detected = 0.0, 1.0, False

        # PSI approximation for categorical
        eps = 1e-6
        ref_pct = ref_aligned / (ref_aligned.sum() + eps)
        cur_pct = cur_aligned / (cur_aligned.sum() + eps)
        ref_pct = np.clip(ref_pct, eps, None)
        cur_pct = np.clip(cur_pct, eps, None)
        psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

        results[feat] = {
            "psi": round(psi, 6),
            "drift_level": classify_drift(psi),
            "chi2_statistic": round(float(chi2), 4),
            "chi2_pvalue": round(float(pval), 6),
            "chi2_drift_detected": drift_detected,
            "reference_distribution": ref_counts.to_dict(),
            "current_distribution": cur_counts.to_dict(),
        }
    return results


def run_drift_analysis(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """Run full drift analysis between reference (train) and current data."""
    log.info(
        "Running drift analysis | Reference: %d rows | Current: %d rows",
        len(reference),
        len(current),
    )

    numeric_results = analyze_numeric_drift(reference, current)
    categorical_results = analyze_categorical_drift(reference, current)

    # Overall drift summary
    all_drift_levels = [v["drift_level"] for v in {**numeric_results, **categorical_results}.values()]
    severe_count = all_drift_levels.count("SEVERE")
    moderate_count = all_drift_levels.count("MODERATE")
    stable_count = all_drift_levels.count("STABLE")

    if severe_count > 0:
        overall_status = "SEVERE — RETRAIN RECOMMENDED"
        action = "Immediate retraining required. Severe drift detected in key features."
    elif moderate_count > 2:
        overall_status = "MODERATE — MONITOR CLOSELY"
        action = "Schedule retraining within 2 weeks. Multiple features showing moderate drift."
    else:
        overall_status = "STABLE — NO ACTION REQUIRED"
        action = "Model is operating within expected distribution bounds."

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": 24,
        "reference_rows": len(reference),
        "current_rows": len(current),
        "numeric_drift": numeric_results,
        "categorical_drift": categorical_results,
        "summary": {
            "overall_status": overall_status,
            "action": action,
            "stable_features": stable_count,
            "moderate_features": moderate_count,
            "severe_features": severe_count,
            "total_features_monitored": len(all_drift_levels),
        },
    }


def main():
    t0 = time.perf_counter()
    log.info("=" * 70)
    log.info("PHASE 24 — DATA DRIFT MONITORING")
    log.info("=" * 70)

    # Load reference (training) data
    train_path = DATA_DIR / "processed" / "train.csv"
    test_path = DATA_DIR / "processed" / "test.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Train data not found at {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found at {test_path}")

    reference = pd.read_csv(train_path)
    # Use test set as "current" distribution for demonstration
    current = pd.read_csv(test_path)

    report = run_drift_analysis(reference, current)
    elapsed = time.perf_counter() - t0

    out_path = REPORTS_DIR / "drift_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, cls=_NumpyEncoder)
    log.info("Drift report saved -> %s", out_path)

    print("\n" + "=" * 80)
    print(f"PHASE 24 — DRIFT MONITORING COMPLETE  |  {elapsed:.2f}s")
    print("=" * 80)
    print(f"  Reference rows : {report['reference_rows']}")
    print(f"  Current rows   : {report['current_rows']}")
    print(f"  Overall status : {report['summary']['overall_status']}")
    print(f"  Action         : {report['summary']['action']}")
    print(f"  Stable         : {report['summary']['stable_features']} features")
    print(f"  Moderate drift : {report['summary']['moderate_features']} features")
    print(f"  Severe drift   : {report['summary']['severe_features']} features")
    print("=" * 80)


if __name__ == "__main__":
    main()
