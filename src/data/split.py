"""
Phase 7 — Train / Validation / Test Split Module
================================================
Performs reproducible, stratified 80/20 train/test splitting and generates
5-fold Stratified Cross-Validation indices on the training set.

Outputs:
  - data/processed/train.csv (5,634 rows)
  - data/processed/test.csv (1,409 rows) - LOCKED until Phase 14
  - data/processed/cv_folds.json (5 stratified fold indices)
  - reports/split_report.json (verification of class ratios)
  - reports/split_distribution.png (visual verification plot)

Leakage Rule (R3):
  The test set is locked. All downstream preprocessing, scaling, imputation,
  encoding, and feature selection must be fitted on training folds only.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Allow running as a script OR imported as a module
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.utils.config import PARAMS, get_path
from src.utils.logging import get_logger

log = get_logger("split")


def execute_stratified_split() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Executes stratified split of interim clean data into train and test sets.
    Generates 5-fold Stratified K-Fold indices on the training set.
    """
    clean_path = get_path("interim_data")
    if not clean_path.exists():
        raise FileNotFoundError(f"Clean interim dataset not found at: {clean_path}")

    log.info("Loading cleaned dataset from: %s", clean_path)
    df = pd.read_csv(clean_path)

    target_col = PARAMS["dataset"]["target_column"]
    test_size = float(PARAMS["split"]["test_size"])
    random_state = int(PARAMS["random_state"])
    n_splits = int(PARAMS["cv"]["n_splits"])

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 1. Stratified 80/20 Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    train_df = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
    test_df = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)

    # 2. 5-Fold Stratified K-Fold on Train Set
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds_info = []
    folds_indices = {}

    for fold_idx, (train_fold_idx, val_fold_idx) in enumerate(skf.split(train_df, train_df[target_col]), 1):
        y_val_fold = train_df.loc[val_fold_idx, target_col]
        val_counts = y_val_fold.value_counts().to_dict()
        val_churn_pct = round(float(y_val_fold.mean()) * 100, 2)

        folds_indices[f"fold_{fold_idx}"] = {
            "train_indices": train_fold_idx.tolist(),
            "val_indices": val_fold_idx.tolist(),
        }
        folds_info.append({
            "fold": fold_idx,
            "train_size": len(train_fold_idx),
            "val_size": len(val_fold_idx),
            "val_churn_count": int(val_counts.get(1, 0)),
            "val_retained_count": int(val_counts.get(0, 0)),
            "val_churn_pct": val_churn_pct,
        })

    # 3. Class Balance Verification
    full_counts = df[target_col].value_counts().to_dict()
    train_counts = train_df[target_col].value_counts().to_dict()
    test_counts = test_df[target_col].value_counts().to_dict()

    summary_report: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "random_state": random_state,
        "test_size": test_size,
        "n_cv_folds": n_splits,
        "original_dataset": {
            "total_rows": len(df),
            "retained_0": int(full_counts.get(0, 0)),
            "churned_1": int(full_counts.get(1, 0)),
            "churn_pct": round(float(df[target_col].mean()) * 100, 2),
            "imbalance_ratio": round(full_counts.get(0, 0) / full_counts.get(1, 1), 4),
        },
        "train_set": {
            "total_rows": len(train_df),
            "retained_0": int(train_counts.get(0, 0)),
            "churned_1": int(train_counts.get(1, 0)),
            "churn_pct": round(float(train_df[target_col].mean()) * 100, 2),
            "imbalance_ratio": round(train_counts.get(0, 0) / train_counts.get(1, 1), 4),
        },
        "test_set": {
            "total_rows": len(test_df),
            "retained_0": int(test_counts.get(0, 0)),
            "churned_1": int(test_counts.get(1, 0)),
            "churn_pct": round(float(test_df[target_col].mean()) * 100, 2),
            "imbalance_ratio": round(test_counts.get(0, 0) / test_counts.get(1, 1), 4),
            "status": "LOCKED until Phase 14",
        },
        "cv_folds": folds_info,
        "integrity_assertions": {
            "stratification_preserved": bool(
                abs(train_df[target_col].mean() - df[target_col].mean()) < 0.001
                and abs(test_df[target_col].mean() - df[target_col].mean()) < 0.001
            ),
            "row_sum_matches": bool(len(train_df) + len(test_df) == len(df)),
            "column_count_matches": bool(train_df.shape[1] == test_df.shape[1] == df.shape[1]),
        },
    }

    # Ensure assertions hold
    assert summary_report["integrity_assertions"]["stratification_preserved"], "Stratification failed!"
    assert summary_report["integrity_assertions"]["row_sum_matches"], "Row sum mismatch!"

    return train_df, test_df, summary_report, folds_indices


def plot_split_verification(report: Dict[str, Any], out_path: Path) -> None:
    """Generates visual comparison verifying identical target class ratios."""
    splits = ["Full Dataset", "Train Set (80%)", "Test Set (20% - Locked)"]
    churn_pcts = [
        report["original_dataset"]["churn_pct"],
        report["train_set"]["churn_pct"],
        report["test_set"]["churn_pct"],
    ]
    retained_pcts = [round(100 - p, 2) for p in churn_pcts]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    # Bar comparison
    x = np.arange(len(splits))
    width = 0.35
    ax1.bar(x - width / 2, retained_pcts, width, label="Retained (0)", color="#4C72B0", edgecolor="black")
    ax1.bar(x + width / 2, churn_pcts, width, label="Churned (1)", color="#DD8452", edgecolor="black")
    ax1.set_ylabel("Class Percentage (%)")
    ax1.set_title("Class Ratio Verification Across Splits", fontsize=12, weight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(splits, fontsize=9, weight="bold")
    ax1.set_ylim(0, 100)
    ax1.legend(loc="upper right")

    for p in ax1.patches:
        h = p.get_height()
        ax1.annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2, h + 1), ha="center", fontsize=9, weight="bold")

    # CV Folds Churn %
    fold_names = [f"Fold {f['fold']}" for f in report["cv_folds"]]
    fold_pcts = [f["val_churn_pct"] for f in report["cv_folds"]]
    ax2.plot(fold_names, fold_pcts, marker="o", color="#2ca02c", linewidth=2, markersize=8)
    ax2.axhline(report["train_set"]["churn_pct"], color="red", linestyle="--", label=f"Train Mean: {report['train_set']['churn_pct']}%")
    ax2.set_ylabel("Validation Fold Churn Rate (%)")
    ax2.set_title("5-Fold Cross-Validation Stratification", fontsize=12, weight="bold")
    ax2.set_ylim(20, 32)
    ax2.legend()
    for name, pct in zip(fold_names, fold_pcts):
        ax2.annotate(f"{pct:.2f}%", (name, pct + 0.5), ha="center", fontsize=9, weight="bold")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    log.info("Saved split distribution plot -> %s", out_path)


def run_split_pipeline() -> Dict[str, Any]:
    """Orchestrates Phase 7 train/test splitting and writes outputs."""
    log.info("Starting Phase 7 — Train / Validation / Test Split")

    train_df, test_df, report, folds_indices = execute_stratified_split()

    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save train and test datasets
    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    log.info("Saved train.csv -> %s (%d rows)", train_path, len(train_df))
    log.info("Saved test.csv  -> %s (%d rows) [LOCKED]", test_path, len(test_df))

    # 2. Save CV fold indices for exact reproducibility across all models
    cv_folds_path = processed_dir / "cv_folds.json"
    with open(cv_folds_path, "w", encoding="utf-8") as f:
        json.dump(folds_indices, f)
    log.info("Saved repeatable CV fold indices -> %s", cv_folds_path)

    # 3. Save report and plot
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "split_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log.info("Saved split report -> %s", report_path)

    plot_split_verification(report, reports_dir / "split_distribution.png")

    return report


if __name__ == "__main__":
    rep = run_split_pipeline()
    print("\n" + "=" * 60)
    print("PHASE 7 — TRAIN / VALIDATION / TEST SPLIT COMPLETED")
    print("=" * 60)
    print(f"Total Rows: {rep['original_dataset']['total_rows']}")
    print(f"Train Rows: {rep['train_set']['total_rows']} (Churn: {rep['train_set']['churn_pct']}%, Ratio: {rep['train_set']['imbalance_ratio']}:1)")
    print(f"Test Rows : {rep['test_set']['total_rows']} (Churn: {rep['test_set']['churn_pct']}%, Ratio: {rep['test_set']['imbalance_ratio']}:1) [LOCKED]")
    print(f"\n5-Fold CV Validation Folds (Train Split):")
    for f in rep["cv_folds"]:
        print(f"  - Fold {f['fold']}: size={f['val_size']}, churn={f['val_churn_count']} ({f['val_churn_pct']}%)")
    print("\nTest Set Lock: TEST SET IS LOCKED UNTIL PHASE 14.")
    print("=" * 60)
