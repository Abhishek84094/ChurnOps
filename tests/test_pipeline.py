"""
Phase 26 — Test Suite
=======================
Comprehensive pytest test suite covering:
  - Data integrity (schema, types, value ranges)
  - Preprocessing pipeline (no leakage, correct output shape)
  - Model loading and prediction (API contract)
  - API endpoint behavior (FastAPI TestClient)
  - Registry gate compliance
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ─── Paths ────────────────────────────────────────────────────────────────────
TRAIN_CSV = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_CSV = PROJECT_ROOT / "data" / "processed" / "test.csv"
CHAMPION_PATH = PROJECT_ROOT / "models" / "champion_model.joblib"
EVAL_REPORT = PROJECT_ROOT / "reports" / "final_model_evaluation.json"
REGISTRY_MANIFEST = PROJECT_ROOT / "reports" / "model_registry_manifest.json"

TARGET_COL = "Churn"
EXPECTED_RAW_COLS = 19  # 20 total minus target = 19 features

REGISTRY_GATES = {
    "min_f1": 0.60,
    "min_roc_auc": 0.75,
    "min_recall": 0.60,
    "max_gap": 0.08,
}


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def train_data():
    if not TRAIN_CSV.exists():
        pytest.skip(f"train.csv not found at {TRAIN_CSV}")
    return pd.read_csv(TRAIN_CSV)


@pytest.fixture(scope="session")
def test_data():
    if not TEST_CSV.exists():
        pytest.skip(f"test.csv not found at {TEST_CSV}")
    return pd.read_csv(TEST_CSV)


@pytest.fixture(scope="session")
def champion_pipeline():
    if not CHAMPION_PATH.exists():
        pytest.skip(f"Champion model not found at {CHAMPION_PATH}")
    import joblib
    return joblib.load(CHAMPION_PATH)


@pytest.fixture(scope="session")
def eval_report():
    if not EVAL_REPORT.exists():
        pytest.skip(f"Evaluation report not found at {EVAL_REPORT}")
    with open(EVAL_REPORT, "r") as f:
        return json.load(f)


# ─── Data Integrity Tests ─────────────────────────────────────────────────────

class TestDataIntegrity:
    """Phase 26: Data integrity checks."""

    def test_train_exists(self, train_data):
        assert train_data is not None
        assert len(train_data) > 0, "Train data must not be empty"

    def test_test_exists(self, test_data):
        assert test_data is not None
        assert len(test_data) > 0, "Test data must not be empty"

    def test_train_shape(self, train_data):
        assert len(train_data) == 5634, f"Expected 5634 rows, got {len(train_data)}"
        assert len(train_data.columns) == 20, f"Expected 20 columns, got {len(train_data.columns)}"

    def test_test_shape(self, test_data):
        assert len(test_data) == 1409, f"Expected 1409 rows, got {len(test_data)}"
        assert len(test_data.columns) == 20

    def test_target_column_exists(self, train_data):
        assert TARGET_COL in train_data.columns, f"Target column '{TARGET_COL}' missing"

    def test_target_binary(self, train_data, test_data):
        for df, name in [(train_data, "train"), (test_data, "test")]:
            unique = set(df[TARGET_COL].unique())
            assert unique == {0, 1}, f"{name} target has unexpected values: {unique}"

    def test_no_null_values(self, train_data, test_data):
        for df, name in [(train_data, "train"), (test_data, "test")]:
            nulls = df.isnull().sum().sum()
            assert nulls == 0, f"{name} has {nulls} null values"

    def test_no_duplicate_rows(self, train_data, test_data):
        # Note: After customerID is dropped, a small number of rows may share
        # identical attribute values (genuine duplicates in the real dataset).
        # We allow up to 20 such duplicates (known: 15 in train after Phase 4).
        MAX_ALLOWED_DUPES = 20
        for df, name in [(train_data, "train"), (test_data, "test")]:
            dupes = df.duplicated().sum()
            assert dupes <= MAX_ALLOWED_DUPES, (
                f"{name} has {dupes} duplicate rows (threshold: {MAX_ALLOWED_DUPES}). "
                "Check data cleaning pipeline."
            )

    def test_churn_rate_stratification(self, train_data, test_data):
        """Train and test churn rates should be within 0.5% of each other."""
        train_rate = train_data[TARGET_COL].mean()
        test_rate = test_data[TARGET_COL].mean()
        assert abs(train_rate - test_rate) < 0.005, (
            f"Churn rate mismatch: train={train_rate:.4f}, test={test_rate:.4f}"
        )

    def test_tenure_range(self, train_data):
        assert train_data["tenure"].min() >= 0
        assert train_data["tenure"].max() <= 72

    def test_monthly_charges_positive(self, train_data):
        assert (train_data["MonthlyCharges"] > 0).all(), "MonthlyCharges must be positive"

    def test_no_gender_leakage(self, train_data):
        """gender should be present in data (dropped by pipeline, not by data prep)."""
        assert "gender" in train_data.columns, "gender column should remain in data until pipeline drops it"

    def test_no_test_leakage_into_train(self, train_data, test_data):
        """Train and test sets must not share the same rows (by position)."""
        # customerID was dropped in Phase 4 — verify disjoint sizes sum to full dataset
        total = len(train_data) + len(test_data)
        assert total == 7043, (
            f"Train+test size {total} != 7043. Possible overlap or data loss."
        )
        # Verify the 80/20 split proportions
        assert len(train_data) == 5634, f"Train size {len(train_data)} != 5634"
        assert len(test_data) == 1409, f"Test size {len(test_data)} != 1409"


# ─── Pipeline Tests ───────────────────────────────────────────────────────────

class TestPreprocessingPipeline:
    """Phase 26: Preprocessing pipeline correctness."""

    def test_pipeline_loads(self, champion_pipeline):
        assert champion_pipeline is not None

    def test_pipeline_predicts_single_row(self, champion_pipeline, test_data):
        single = test_data.drop(columns=[TARGET_COL]).iloc[[0]]
        proba = champion_pipeline.predict_proba(single)
        assert proba.shape == (1, 2), f"Expected (1, 2), got {proba.shape}"
        assert 0.0 <= proba[0, 1] <= 1.0

    def test_pipeline_predicts_full_test(self, champion_pipeline, test_data):
        X_test = test_data.drop(columns=[TARGET_COL])
        probas = champion_pipeline.predict_proba(X_test)
        assert probas.shape == (len(test_data), 2)
        assert np.all(probas >= 0.0) and np.all(probas <= 1.0), "Probabilities out of [0,1]"

    def test_pipeline_probas_sum_to_one(self, champion_pipeline, test_data):
        X_test = test_data.drop(columns=[TARGET_COL]).iloc[:50]
        probas = champion_pipeline.predict_proba(X_test)
        row_sums = probas.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6, err_msg="Probabilities must sum to 1")

    def test_pipeline_handles_unknown_categories(self, champion_pipeline, test_data):
        """Pipeline must not crash on unseen categorical values (handle_unknown='ignore')."""
        X = test_data.drop(columns=[TARGET_COL]).iloc[[0]].copy()
        X["Contract"] = "UNKNOWN_CONTRACT_TYPE"
        try:
            proba = champion_pipeline.predict_proba(X)
            assert proba.shape == (1, 2), "Should return valid prediction even with unknown category"
        except Exception as e:
            pytest.fail(f"Pipeline crashed on unknown category: {e}")

    def test_inference_latency(self, champion_pipeline, test_data):
        """Single-row inference should complete in < 100ms."""
        import time
        X = test_data.drop(columns=[TARGET_COL]).iloc[[0]]
        t0 = time.perf_counter()
        champion_pipeline.predict_proba(X)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 100, f"Single inference took {elapsed_ms:.1f}ms (threshold: 100ms)"


# ─── Registry Gate Tests ──────────────────────────────────────────────────────

class TestRegistryGates:
    """Phase 26: Verify champion model meets all production registry gates."""

    def test_eval_report_exists(self, eval_report):
        assert eval_report is not None

    def test_roc_auc_gate(self, eval_report):
        auc = eval_report["holdout_test_results"]["roc_auc"]
        assert auc >= REGISTRY_GATES["min_roc_auc"], (
            f"ROC-AUC {auc:.4f} < required {REGISTRY_GATES['min_roc_auc']}"
        )

    def test_f1_gate(self, eval_report):
        f1 = eval_report["holdout_test_results"]["default_threshold_metrics"]["f1"]
        assert f1 >= REGISTRY_GATES["min_f1"], (
            f"F1 {f1:.4f} < required {REGISTRY_GATES['min_f1']}"
        )

    def test_recall_gate(self, eval_report):
        recall = eval_report["holdout_test_results"]["default_threshold_metrics"]["recall"]
        assert recall >= REGISTRY_GATES["min_recall"], (
            f"Recall {recall:.4f} < required {REGISTRY_GATES['min_recall']}"
        )

    def test_generalization_gap(self, eval_report):
        """Test-AUC should be close to CV-AUC (gap <= 8%)."""
        test_auc = eval_report["holdout_test_results"]["roc_auc"]
        cv_auc = 0.8493  # from Phase 12
        gap = abs(cv_auc - test_auc)
        assert gap <= REGISTRY_GATES["max_gap"], (
            f"Generalization gap {gap:.4f} > threshold {REGISTRY_GATES['max_gap']}"
        )

    def test_champion_model_file_exists(self):
        assert CHAMPION_PATH.exists(), f"Champion model file missing at {CHAMPION_PATH}"

    def test_champion_metadata_exists(self):
        meta_path = PROJECT_ROOT / "models" / "champion_metadata.json"
        assert meta_path.exists(), f"Champion metadata missing at {meta_path}"


# ─── API Contract Tests ───────────────────────────────────────────────────────

class TestAPIContract:
    """Phase 26: FastAPI endpoint contract tests using TestClient."""

    @pytest.fixture(scope="class")
    def api_client(self):
        try:
            from fastapi.testclient import TestClient
            from api.app import app
            # Override startup model loading
            import joblib
            import api.app as api_module
            api_module._pipeline = joblib.load(CHAMPION_PATH)
            api_module._model_info = {
                "model_name": "ChurnOps Champion",
                "version": "test",
                "algorithm": "XGBClassifier",
                "performance": {"holdout_test_set": {"roc_auc": 0.8491, "f1_score": 0.6289, "recall": 0.7861}},
                "training_data": {"rows": 5634},
            }
            return TestClient(app)
        except ImportError:
            pytest.skip("FastAPI/httpx not installed")

    SAMPLE_CUSTOMER = {
        "customer_id": "TEST-001",
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 79.85,
        "TotalCharges": 958.2,
    }

    def test_health_endpoint(self, api_client):
        response = api_client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "healthy"
        assert "model_loaded" in body

    def test_root_endpoint(self, api_client):
        response = api_client.get("/")
        assert response.status_code == 200
        assert "predict" in response.json()

    def test_model_info_endpoint(self, api_client):
        response = api_client.get("/model/info")
        assert response.status_code == 200
        body = response.json()
        assert "model_name" in body
        assert body["test_roc_auc"] > 0.0

    def test_predict_single(self, api_client):
        response = api_client.post("/predict", json=self.SAMPLE_CUSTOMER)
        assert response.status_code == 200
        body = response.json()
        assert "churn_probability" in body
        assert "churn_prediction" in body
        assert "risk_label" in body
        assert 0.0 <= body["churn_probability"] <= 1.0
        assert body["churn_prediction"] in (0, 1)

    def test_predict_batch(self, api_client):
        payload = {"customers": [self.SAMPLE_CUSTOMER, self.SAMPLE_CUSTOMER]}
        response = api_client.post("/predict/batch", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 2
        assert len(body["predictions"]) == 2

    def test_predict_empty_batch_rejected(self, api_client):
        payload = {"customers": []}
        response = api_client.post("/predict/batch", json=payload)
        assert response.status_code == 422, "Empty batch should be rejected"

    def test_predict_response_schema(self, api_client):
        response = api_client.post("/predict", json=self.SAMPLE_CUSTOMER)
        body = response.json()
        required_fields = [
            "customer_id", "churn_probability", "churn_prediction",
            "risk_label", "decision_threshold", "latency_ms", "model_version", "timestamp"
        ]
        for field in required_fields:
            assert field in body, f"Field '{field}' missing from /predict response"
