"""
Phase 18 — Kubeflow Pipeline Design
=====================================
Defines the ChurnOps end-to-end ML pipeline using the KFP v2 SDK.
This pipeline encapsulates data ingestion through model evaluation as
containerized, reusable KFP components suitable for execution on
Google Cloud Vertex AI Pipelines or any KFP-compatible cluster.

Usage:
    python src/pipelines/kubeflow_pipeline.py
    # → generates deploy/churnops_pipeline.yaml (compiled pipeline spec)
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import kfp
    from kfp import dsl
    from kfp.dsl import Dataset, Input, Output, Artifact, Model, Metrics
    KFP_AVAILABLE = True
except ImportError:
    KFP_AVAILABLE = False

from src.utils.logging import get_logger

log = get_logger("kubeflow_pipeline")

DEPLOY_DIR = Path("deploy")
PIPELINE_YAML = DEPLOY_DIR / "churnops_pipeline.yaml"
BASE_IMAGE = "python:3.11-slim"
CHURNOPS_IMAGE = "churnops-api:1.0.0"  # built via Dockerfile


# ─── KFP Component Definitions ────────────────────────────────────────────────

def _make_components():
    """Define KFP components (only if KFP is available)."""
    if not KFP_AVAILABLE:
        return None, None, None, None, None, None, None, None

    @dsl.component(
        base_image=BASE_IMAGE,
        packages_to_install=["pandas==2.1.4", "scikit-learn==1.4.2"],
    )
    def ingest_and_validate(
        raw_csv_path: str,
        dataset: Output[Dataset],
        metadata: Output[Artifact],
    ):
        """Phase 1-2: Ingest raw CSV and run validation checks."""
        import json
        import hashlib
        import pandas as pd

        df = pd.read_csv(raw_csv_path)
        assert df.shape == (7043, 21), f"Unexpected shape: {df.shape}"
        assert "Churn" in df.columns, "Target column 'Churn' missing"
        assert df.duplicated().sum() == 0, "Duplicate rows found"

        df.to_csv(dataset.path, index=False)

        sha256 = hashlib.sha256(open(raw_csv_path, "rb").read()).hexdigest()
        meta = {
            "rows": len(df),
            "columns": list(df.columns),
            "sha256": sha256,
            "churn_rate": float(df["Churn"].map({"Yes": 1, "No": 0}).mean()),
        }
        with open(metadata.path, "w") as f:
            json.dump(meta, f)

    @dsl.component(
        base_image=BASE_IMAGE,
        packages_to_install=["pandas==2.1.4", "scikit-learn==1.4.2"],
    )
    def clean_and_engineer(
        raw_dataset: Input[Dataset],
        engineered_dataset: Output[Dataset],
    ):
        """Phase 4-5: Data cleaning and feature engineering."""
        import pandas as pd

        df = pd.read_csv(raw_dataset.path)

        # Clean
        df["TotalCharges"] = df["TotalCharges"].replace(" ", "0.0").astype(float)
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
        df.drop(columns=["customerID"], errors="ignore", inplace=True)

        # Feature engineering
        df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
        df["has_fiber_optic"] = (df["InternetService"] == "Fiber optic").astype(int)
        df["is_auto_pay"] = df["PaymentMethod"].str.contains("automatic", case=False).astype(int)
        df["has_family"] = ((df["Partner"] == "Yes") | (df["Dependents"] == "Yes")).astype(int)

        df.to_csv(engineered_dataset.path, index=False)

    @dsl.component(
        base_image=BASE_IMAGE,
        packages_to_install=["pandas==2.1.4", "scikit-learn==1.4.2"],
    )
    def split_data(
        engineered_dataset: Input[Dataset],
        train_dataset: Output[Dataset],
        test_dataset: Output[Dataset],
    ):
        """Phase 7: Stratified train/test split."""
        import pandas as pd
        from sklearn.model_selection import train_test_split

        df = pd.read_csv(engineered_dataset.path)
        train, test = train_test_split(
            df, test_size=0.20, random_state=42, stratify=df["Churn"]
        )
        train.to_csv(train_dataset.path, index=False)
        test.to_csv(test_dataset.path, index=False)

    @dsl.component(
        base_image=BASE_IMAGE,
        packages_to_install=[
            "pandas==2.1.4", "scikit-learn==1.4.2",
            "xgboost==2.0.3", "joblib==1.3.2",
        ],
    )
    def train_and_tune(
        train_dataset: Input[Dataset],
        model: Output[Model],
        metrics: Output[Metrics],
        n_estimators: int = 200,
        max_depth: int = 2,
        learning_rate: float = 0.08,
    ):
        """Phase 9-12: Train baseline and tune XGBoost champion."""
        import pandas as pd
        import joblib
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.impute import SimpleImputer
        from xgboost import XGBClassifier
        from sklearn.model_selection import cross_val_score

        df = pd.read_csv(train_dataset.path)
        X = df.drop(columns=["Churn"])
        y = df["Churn"]

        numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
        cat_cols = [c for c in X.select_dtypes("object").columns]

        preprocessor = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), numeric_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols),
        ], remainder="passthrough")

        clf = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            scale_pos_weight=2.5,
            reg_alpha=1.0,
            reg_lambda=20.0,
            random_state=42,
            eval_metric="logloss",
        )

        pipeline = Pipeline([("prep", preprocessor), ("clf", clf)])
        cv_auc = cross_val_score(pipeline, X, y, cv=5, scoring="roc_auc").mean()
        pipeline.fit(X, y)

        joblib.dump(pipeline, model.path)
        metrics.log_metric("cv_roc_auc", float(cv_auc))

    @dsl.component(
        base_image=BASE_IMAGE,
        packages_to_install=[
            "pandas==2.1.4", "scikit-learn==1.4.2",
            "xgboost==2.0.3", "joblib==1.3.2",
        ],
    )
    def evaluate_champion(
        test_dataset: Input[Dataset],
        model: Input[Model],
        eval_metrics: Output[Metrics],
    ):
        """Phase 14: Holdout test evaluation and gate check."""
        import pandas as pd
        import joblib
        from sklearn.metrics import roc_auc_score, f1_score, recall_score

        df = pd.read_csv(test_dataset.path)
        X = df.drop(columns=["Churn"])
        y = df["Churn"].values

        pipeline = joblib.load(model.path)
        probas = pipeline.predict_proba(X)[:, 1]
        preds = (probas >= 0.48).astype(int)

        auc = float(roc_auc_score(y, probas))
        f1 = float(f1_score(y, preds))
        recall = float(recall_score(y, preds))

        eval_metrics.log_metric("test_roc_auc", auc)
        eval_metrics.log_metric("test_f1", f1)
        eval_metrics.log_metric("test_recall", recall)

        assert auc >= 0.75, f"AUC gate failed: {auc:.4f} < 0.75"
        assert f1 >= 0.60, f"F1 gate failed: {f1:.4f} < 0.60"
        assert recall >= 0.60, f"Recall gate failed: {recall:.4f} < 0.60"

    return ingest_and_validate, clean_and_engineer, split_data, train_and_tune, evaluate_champion


# ─── Pipeline Definition ───────────────────────────────────────────────────────

def build_pipeline(raw_csv_path: str = "gs://churnops-data/raw/telco_churn.csv"):
    """Build and compile the ChurnOps KFP pipeline."""
    if not KFP_AVAILABLE:
        log.warning("KFP not installed. Cannot compile pipeline.")
        return None

    (
        ingest_and_validate,
        clean_and_engineer,
        split_data,
        train_and_tune,
        evaluate_champion,
    ) = _make_components()

    @dsl.pipeline(
        name="ChurnOps MLOps Pipeline",
        description=(
            "End-to-end customer churn prediction pipeline: "
            "Ingest → Clean → Engineer → Split → Train/Tune → Evaluate → Register"
        ),
    )
    def churnops_pipeline(
        raw_csv_path: str = raw_csv_path,
        n_estimators: int = 200,
        max_depth: int = 2,
        learning_rate: float = 0.08,
    ):
        ingest_task = ingest_and_validate(raw_csv_path=raw_csv_path)
        clean_task = clean_and_engineer(raw_dataset=ingest_task.outputs["dataset"])
        split_task = split_data(engineered_dataset=clean_task.outputs["engineered_dataset"])
        train_task = train_and_tune(
            train_dataset=split_task.outputs["train_dataset"],
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
        )
        evaluate_champion(
            test_dataset=split_task.outputs["test_dataset"],
            model=train_task.outputs["model"],
        )

    return churnops_pipeline


def main():
    """Compile the pipeline to YAML and save to deploy/."""
    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)

    if not KFP_AVAILABLE:
        log.warning("kfp not installed. Generating pipeline spec stub.")
        # Write a human-readable stub for documentation purposes
        stub = {
            "name": "ChurnOps MLOps Pipeline",
            "description": "End-to-end churn prediction pipeline (KFP v2 spec)",
            "note": "Install kfp>=2.7.0 to compile. Run: pip install kfp==2.7.0",
            "components": [
                "ingest_and_validate",
                "clean_and_engineer",
                "split_data",
                "train_and_tune",
                "evaluate_champion",
            ],
            "platform": "Google Cloud Vertex AI Pipelines / KFP-compatible cluster",
            "compile_command": "python src/pipelines/kubeflow_pipeline.py",
        }
        import json
        stub_path = DEPLOY_DIR / "churnops_pipeline_stub.json"
        with open(stub_path, "w") as f:
            json.dump(stub, f, indent=2)
        print(f"Pipeline stub saved -> {stub_path}")
        print("Install kfp with: pip install kfp==2.7.0 to compile the full YAML")
        return

    pipeline_fn = build_pipeline()
    kfp.compiler.Compiler().compile(
        pipeline_func=pipeline_fn,
        package_path=str(PIPELINE_YAML),
    )
    log.info("Pipeline compiled -> %s", PIPELINE_YAML)
    print(f"\nKubeflow pipeline compiled -> {PIPELINE_YAML}")
    print("Deploy to Vertex AI with:")
    print("  gcloud ai custom-jobs create --region=us-central1 \\")
    print("    --display-name=churnops \\")
    print(f"    --pipeline-spec={PIPELINE_YAML}")


if __name__ == "__main__":
    main()
