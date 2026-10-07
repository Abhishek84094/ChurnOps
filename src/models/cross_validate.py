"""
Phase 10 — Cross-Validation Deep-Dive
======================================
Detailed analysis of the 5-fold CV results from Phase 9.

Analyses produced:
  1. Learning Curves      — train vs. val F1 as training size grows.
  2. Bias-Variance        — mean train/val gap per model (over-/underfitting diagnosis).
  3. Per-fold metric distributions — violin plots of F1, AUC, Recall per model.
  4. Calibration Curves   — predicted probability vs. actual positive rate.
  5. Threshold Sensitivity — F1, Precision, Recall at decision thresholds 0.1–0.9.

Leakage Rule (R3):
  All analysis uses only data/processed/train.csv and cv_folds.json.
  test.csv is NEVER accessed.

Outputs:
  reports/cv_learning_curves.png
  reports/cv_fold_distributions.png
  reports/cv_calibration_curves.png
  reports/cv_threshold_sensitivity.png
  reports/cv_analysis_report.json
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

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script mode
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from src.data.preprocessing import build_full_pipeline
from src.utils.config import PARAMS
from src.utils.logging import get_logger

log = get_logger("cross_validate")

RANDOM_STATE: int = PARAMS["random_state"]
TARGET: str = PARAMS["dataset"]["target_column"]
GAP_THRESHOLD: float = PARAMS["diagnosis"]["gap_threshold"]
LOW_F1: float = PARAMS["diagnosis"]["low_performance_f1"]
REPORTS_DIR = Path("reports")

MODEL_COLORS: Dict[str, str] = {
    "DummyClassifier":    "#6B7280",
    "LogisticRegression": "#3B82F6",
    "RandomForest":       "#10B981",
    "XGBoost":            "#F59E0B",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_spw(y: pd.Series) -> float:
    neg = int((y == 0).sum())
    pos = int((y == 1).sum())
    return float(neg / pos) if pos > 0 else 1.0


def _build_pipelines(y_ref: pd.Series) -> Dict[str, Any]:
    spw = _compute_spw(y_ref)
    return {
        "DummyClassifier": build_full_pipeline(
            estimator=DummyClassifier(strategy="stratified", random_state=RANDOM_STATE),
            scale_numeric=True,
        ),
        "LogisticRegression": build_full_pipeline(
            estimator=LogisticRegression(
                class_weight="balanced", max_iter=1000,
                solver="lbfgs", random_state=RANDOM_STATE,
            ),
            scale_numeric=True,
        ),
        "RandomForest": build_full_pipeline(
            estimator=RandomForestClassifier(
                n_estimators=200, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=-1,
            ),
            scale_numeric=False,
        ),
        "XGBoost": build_full_pipeline(
            estimator=XGBClassifier(
                n_estimators=200, scale_pos_weight=spw,
                eval_metric="logloss", random_state=RANDOM_STATE,
                n_jobs=-1, verbosity=0,
            ),
            scale_numeric=False,
        ),
    }


def _fold_splits(
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: Dict[str, Dict[str, List[int]]],
) -> List[Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]]:
    """Returns list of (X_tr, y_tr, X_val, y_val) for each sorted fold."""
    splits = []
    for fold_name in sorted(cv_folds.keys()):
        fold = cv_folds[fold_name]
        splits.append((
            X.iloc[fold["train_indices"]].reset_index(drop=True),
            y.iloc[fold["train_indices"]].reset_index(drop=True),
            X.iloc[fold["val_indices"]].reset_index(drop=True),
            y.iloc[fold["val_indices"]].reset_index(drop=True),
        ))
    return splits


# ---------------------------------------------------------------------------
# Analysis 1: Bias-Variance Gap (fast — run first for quick logging)
# ---------------------------------------------------------------------------

def compute_bias_variance(
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: Dict[str, Any],
    model_names: List[str],
) -> Dict[str, Dict[str, Any]]:
    """
    Fits each model on the full training fold and records train-F1 vs val-F1.
    Returns {model: {mean_train_f1, mean_val_f1, gap, diagnosis}}.
    """
    splits = _fold_splits(X, y, cv_folds)
    bv: Dict[str, Dict[str, Any]] = {}

    for name in model_names:
        log.info("  Bias-variance: %s", name)
        train_f1s, val_f1s = [], []

        for X_tr, y_tr, X_val, y_val in splits:
            pipes = _build_pipelines(y_tr)
            pipe  = pipes[name]
            if name == "XGBoost":
                pipe.named_steps["model"].set_params(scale_pos_weight=_compute_spw(y_tr))

            pipe.fit(X_tr, y_tr)
            train_f1s.append(f1_score(y_tr,  pipe.predict(X_tr),  zero_division=0))
            val_f1s.append(  f1_score(y_val, pipe.predict(X_val), zero_division=0))

        mean_train = float(np.mean(train_f1s))
        mean_val   = float(np.mean(val_f1s))
        gap        = round(mean_train - mean_val, 4)

        if mean_val < LOW_F1:
            diagnosis = "underfitting"
        elif gap > GAP_THRESHOLD:
            diagnosis = "overfitting"
        else:
            diagnosis = "good_fit"

        bv[name] = {
            "mean_train_f1": round(mean_train, 4),
            "mean_val_f1":   round(mean_val, 4),
            "gap":           gap,
            "diagnosis":     diagnosis,
        }
        log.info(
            "    %-22s train=%.4f val=%.4f gap=%.4f -> %s",
            name, mean_train, mean_val, gap, diagnosis,
        )

    return bv


# ---------------------------------------------------------------------------
# Analysis 2: Learning Curves
# ---------------------------------------------------------------------------

def compute_learning_curves(
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: Dict[str, Any],
    model_names: List[str],
    train_fracs: List[float] = None,
) -> Dict[str, Dict[str, List[float]]]:
    """
    For each model, trains on growing subsets of each fold's training split
    and records mean train-F1 and val-F1 across folds.
    """
    if train_fracs is None:
        train_fracs = [0.10, 0.25, 0.40, 0.60, 0.80, 1.00]

    splits = _fold_splits(X, y, cv_folds)
    results: Dict[str, Dict[str, List[float]]] = {}

    for name in model_names:
        log.info("  Learning curve: %s", name)
        frac_train: Dict[float, List[float]] = {f: [] for f in train_fracs}
        frac_val:   Dict[float, List[float]] = {f: [] for f in train_fracs}

        for X_tr, y_tr, X_val, y_val in splits:
            n_total = len(X_tr)
            for frac in train_fracs:
                n = max(50, int(n_total * frac))
                X_sub = X_tr.iloc[:n]
                y_sub = y_tr.iloc[:n]

                sub_pipes = _build_pipelines(y_sub)
                pipe = sub_pipes[name]
                if name == "XGBoost":
                    pipe.named_steps["model"].set_params(
                        scale_pos_weight=_compute_spw(y_sub))

                pipe.fit(X_sub, y_sub)
                frac_train[frac].append(
                    f1_score(y_sub, pipe.predict(X_sub), zero_division=0))
                frac_val[frac].append(
                    f1_score(y_val, pipe.predict(X_val), zero_division=0))

        results[name] = {
            "fracs":    train_fracs,
            "train_f1": [round(float(np.mean(frac_train[f])), 4) for f in train_fracs],
            "val_f1":   [round(float(np.mean(frac_val[f])),   4) for f in train_fracs],
        }

    return results


def plot_learning_curves(
    lc_data: Dict[str, Dict[str, List[float]]],
    n_total_train: int,
) -> None:
    names = list(lc_data.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4), sharey=True)

    for ax, name in zip(axes, names):
        data  = lc_data[name]
        fracs = data["fracs"]
        sizes = [max(50, int(n_total_train * f)) for f in fracs]
        color = MODEL_COLORS.get(name, "steelblue")

        ax.plot(sizes, data["train_f1"], "o-",  color=color, lw=2, label="Train F1")
        ax.plot(sizes, data["val_f1"],   "s--", color=color, lw=2,
                alpha=0.7, label="Val F1")
        ax.fill_between(sizes, data["train_f1"], data["val_f1"],
                        alpha=0.12, color=color)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Training samples", fontsize=10)
        ax.set_ylabel("F1 Score", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.05)

    plt.suptitle("Phase 10 — Learning Curves (5-Fold CV avg)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = REPORTS_DIR / "cv_learning_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved: %s", out)


# ---------------------------------------------------------------------------
# Analysis 3: Per-fold metric violin plots
# ---------------------------------------------------------------------------

def plot_fold_distributions(
    fold_results: Dict[str, List[Dict[str, float]]],
) -> None:
    metrics = ["f1", "roc_auc", "recall", "precision"]
    titles  = ["F1", "ROC-AUC", "Recall", "Precision"]
    names   = [n for n in ["DummyClassifier", "LogisticRegression",
                            "RandomForest", "XGBoost"]
               if n in fold_results]

    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 5))
    for ax, metric, title in zip(axes, metrics, titles):
        data_plot = [[fd[metric] for fd in fold_results[n]] for n in names]
        parts = ax.violinplot(data_plot, showmeans=True, showmedians=False)
        for pc, name in zip(parts["bodies"], names):
            pc.set_facecolor(MODEL_COLORS.get(name, "gray"))
            pc.set_alpha(0.65)
        ax.set_xticks(range(1, len(names) + 1))
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(title, fontsize=10)
        ax.grid(axis="y", alpha=0.35)

    plt.suptitle("Phase 10 — Per-Fold Metric Distributions (5-Fold CV)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = REPORTS_DIR / "cv_fold_distributions.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved: %s", out)


# ---------------------------------------------------------------------------
# Analysis 4: Calibration Curves (last fold)
# ---------------------------------------------------------------------------

def plot_calibration_curves(
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: Dict[str, Any],
    model_names: List[str],
) -> None:
    last_fold_name = sorted(cv_folds.keys())[-1]
    fold  = cv_folds[last_fold_name]
    X_tr  = X.iloc[fold["train_indices"]].reset_index(drop=True)
    y_tr  = y.iloc[fold["train_indices"]].reset_index(drop=True)
    X_val = X.iloc[fold["val_indices"]].reset_index(drop=True)
    y_val = y.iloc[fold["val_indices"]].reset_index(drop=True)

    pipes = _build_pipelines(y_tr)
    for name, pipe in pipes.items():
        if name == "XGBoost":
            pipe.named_steps["model"].set_params(scale_pos_weight=_compute_spw(y_tr))
        pipe.fit(X_tr, y_tr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Perfect calibration")

    for name in model_names:
        if name == "DummyClassifier":
            continue
        pipe = pipes[name]
        try:
            y_prob = pipe.predict_proba(X_val)[:, 1]
        except AttributeError:
            continue
        frac_pos, mean_pred = calibration_curve(y_val, y_prob, n_bins=8)
        ax.plot(mean_pred, frac_pos, "o-", lw=2,
                color=MODEL_COLORS[name], label=name)

    ax.set_xlabel("Mean predicted probability", fontsize=11)
    ax.set_ylabel("Fraction of positives", fontsize=11)
    ax.set_title("Phase 10 — Calibration Curves (Fold 5)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = REPORTS_DIR / "cv_calibration_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved: %s", out)


# ---------------------------------------------------------------------------
# Analysis 5: Threshold Sensitivity (last fold)
# ---------------------------------------------------------------------------

def compute_threshold_sensitivity(
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: Dict[str, Any],
    model_names: List[str],
    thresholds: List[float] = None,
) -> Dict[str, Dict[str, List]]:
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.10, 0.91, 0.05).tolist()]

    last_fold_name = sorted(cv_folds.keys())[-1]
    fold  = cv_folds[last_fold_name]
    X_tr  = X.iloc[fold["train_indices"]].reset_index(drop=True)
    y_tr  = y.iloc[fold["train_indices"]].reset_index(drop=True)
    X_val = X.iloc[fold["val_indices"]].reset_index(drop=True)
    y_val = y.iloc[fold["val_indices"]].reset_index(drop=True)

    pipes = _build_pipelines(y_tr)
    for name, pipe in pipes.items():
        if name == "XGBoost":
            pipe.named_steps["model"].set_params(scale_pos_weight=_compute_spw(y_tr))
        pipe.fit(X_tr, y_tr)

    results: Dict[str, Dict[str, List]] = {}

    for name in model_names:
        if name == "DummyClassifier":
            continue
        pipe = pipes[name]
        try:
            y_prob = pipe.predict_proba(X_val)[:, 1]
        except AttributeError:
            continue

        f1s, precs, recs = [], [], []
        for t in thresholds:
            y_pred_t = (y_prob >= t).astype(int)
            f1s.append(round(f1_score(y_val, y_pred_t, zero_division=0), 4))
            precs.append(round(precision_score(y_val, y_pred_t, zero_division=0), 4))
            recs.append(round(recall_score(y_val, y_pred_t, zero_division=0), 4))

        best_idx = int(np.argmax(f1s))
        log.info(
            "  %-22s best_t=%.2f  F1=%.4f  Prec=%.4f  Rec=%.4f",
            name, thresholds[best_idx], f1s[best_idx],
            precs[best_idx], recs[best_idx],
        )
        results[name] = {
            "thresholds": thresholds,
            "f1":         f1s,
            "precision":  precs,
            "recall":     recs,
        }

    return results


def plot_threshold_sensitivity(
    ts_data: Dict[str, Dict[str, List]],
) -> None:
    names = list(ts_data.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(5.5 * len(names), 4.5), sharey=True)
    if len(names) == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        data  = ts_data[name]
        thrs  = data["thresholds"]
        color = MODEL_COLORS.get(name, "steelblue")

        ax.plot(thrs, data["f1"],        color=color, lw=2.5, label="F1")
        ax.plot(thrs, data["precision"], color=color, lw=1.5,
                ls="--", alpha=0.8, label="Precision")
        ax.plot(thrs, data["recall"],    color=color, lw=1.5,
                ls=":",  alpha=0.8, label="Recall")

        best_idx = int(np.argmax(data["f1"]))
        ax.axvline(thrs[best_idx], color="red", ls="--", alpha=0.6,
                   label=f"best t={thrs[best_idx]:.2f}")

        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Decision Threshold", fontsize=10)
        ax.set_ylabel("Score", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.05)

    plt.suptitle("Phase 10 — Threshold Sensitivity (Fold 5 val)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = REPORTS_DIR / "cv_threshold_sensitivity.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved: %s", out)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_cv_analysis() -> Dict[str, Any]:
    """Main entry point for Phase 10."""
    log.info("=" * 60)
    log.info("Phase 10 — Cross-Validation Deep-Dive")
    log.info("=" * 60)

    train_path = Path("data/processed/train.csv")
    folds_path = Path("data/processed/cv_folds.json")

    df_train = pd.read_csv(train_path)
    with open(folds_path, "r", encoding="utf-8") as f:
        cv_folds = json.load(f)

    X = df_train.drop(columns=[TARGET])
    y = df_train[TARGET]

    log.info("Train: %d rows  |  Folds: %d", len(df_train), len(cv_folds))
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model_names = ["DummyClassifier", "LogisticRegression", "RandomForest", "XGBoost"]

    # 1. Bias-Variance (fast)
    log.info("\n--- Analysis 1: Bias-Variance Gap ---")
    bv = compute_bias_variance(X, y, cv_folds, model_names)

    # 2. Learning Curves
    log.info("\n--- Analysis 2: Learning Curves ---")
    lc = compute_learning_curves(X, y, cv_folds, model_names)
    plot_learning_curves(lc, n_total_train=len(X))

    # 3. Violin plots from Phase 9 fold results
    phase9_path = Path("reports/baseline_cv_results.json")
    if phase9_path.exists():
        log.info("\n--- Analysis 3: Per-Fold Distributions ---")
        with open(phase9_path, "r", encoding="utf-8") as f:
            p9 = json.load(f)
        fold_results_for_violin = p9.get("fold_results", {})
        if fold_results_for_violin:
            plot_fold_distributions(fold_results_for_violin)
    else:
        log.warning("Phase 9 baseline_cv_results.json not found; skipping violin plots.")

    # 4. Calibration
    log.info("\n--- Analysis 4: Calibration Curves ---")
    plot_calibration_curves(X, y, cv_folds, model_names)

    # 5. Threshold sensitivity
    log.info("\n--- Analysis 5: Threshold Sensitivity ---")
    ts = compute_threshold_sensitivity(X, y, cv_folds, model_names)
    plot_threshold_sensitivity(ts)

    # Compile best thresholds
    best_thresholds: Dict[str, Any] = {}
    for name, data in ts.items():
        best_idx = int(np.argmax(data["f1"]))
        best_thresholds[name] = {
            "best_threshold":    data["thresholds"][best_idx],
            "f1_at_best":        data["f1"][best_idx],
            "precision_at_best": data["precision"][best_idx],
            "recall_at_best":    data["recall"][best_idx],
        }

    report: Dict[str, Any] = {
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "phase":           10,
        "train_rows":      len(df_train),
        "n_folds":         len(cv_folds),
        "gap_threshold":   GAP_THRESHOLD,
        "bias_variance":   bv,
        "best_thresholds": best_thresholds,
        "learning_curves": {
            name: {"fracs": v["fracs"], "train_f1": v["train_f1"], "val_f1": v["val_f1"]}
            for name, v in lc.items()
        },
    }

    out_path = REPORTS_DIR / "cv_analysis_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log.info("Saved report -> %s", out_path)

    return report


if __name__ == "__main__":
    t0 = time.perf_counter()
    report = run_cv_analysis()
    elapsed = time.perf_counter() - t0

    print("\n" + "=" * 70)
    print("PHASE 10 — CROSS-VALIDATION DEEP-DIVE  |  Results")
    print("=" * 70)

    print(f"\n{'Model':<24} {'TrainF1':>9} {'ValF1':>9} {'Gap':>8}  Diagnosis")
    print("-" * 70)
    for name, bv in report["bias_variance"].items():
        print(f"{name:<24} {bv['mean_train_f1']:>9.4f} {bv['mean_val_f1']:>9.4f} "
              f"{bv['gap']:>8.4f}  {bv['diagnosis']}")

    print(f"\n{'Model':<24} {'BestThresh':>11} {'F1':>8} {'Prec':>8} {'Rec':>8}")
    print("-" * 70)
    for name, bt in report["best_thresholds"].items():
        print(f"{name:<24} {bt['best_threshold']:>11.2f} "
              f"{bt['f1_at_best']:>8.4f} {bt['precision_at_best']:>8.4f} "
              f"{bt['recall_at_best']:>8.4f}")

    print(f"\nTotal time: {elapsed:.1f}s")
    print("=" * 70)
