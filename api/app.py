"""
Phase 22 — FastAPI Inference Server
=====================================
Production-grade REST API for real-time churn prediction.
Serves the champion XGBoost pipeline with:
  - /predict endpoint (single customer)
  - /predict/batch endpoint (list of customers)
  - /health endpoint
  - /model/info endpoint
  - Structured request/response schemas
  - Logging and error handling
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.schemas import (
    CustomerFeatures,
    BatchPredictRequest,
    PredictResponse,
    BatchPredictResponse,
    HealthResponse,
    ModelInfoResponse,
)

# ── Constants ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
CHAMPION_PATH = MODELS_DIR / "champion_model.joblib"
MODEL_CARD_PATH = MODELS_DIR / "model_card.json"
DECISION_THRESHOLD = 0.48  # Optimal threshold from Phase 14

# ── App Init ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ChurnOps — Churn Prediction API",
    description=(
        "Real-time customer churn prediction using a Tuned XGBoost model. "
        "Test AUC: 0.8491 | Test F1: 0.6289 | Test Recall: 0.7861"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model Loading ─────────────────────────────────────────────────────────────
_pipeline = None
_model_info: Dict[str, Any] = {}
_startup_time = datetime.now(timezone.utc).isoformat()
_predict_count = 0


def load_model():
    """Load champion pipeline from disk."""
    global _pipeline, _model_info
    if not CHAMPION_PATH.exists():
        raise FileNotFoundError(
            f"Champion model not found at {CHAMPION_PATH}. "
            "Run Phase 14 (evaluate.py) to generate it."
        )
    _pipeline = joblib.load(CHAMPION_PATH)

    # Load model card if available
    if MODEL_CARD_PATH.exists():
        with open(MODEL_CARD_PATH, "r") as f:
            _model_info = json.load(f)
    else:
        _model_info = {
            "model_name": "ChurnOps Champion — Tuned XGBoost",
            "version": "latest",
            "performance": {
                "holdout_test_set": {
                    "roc_auc": 0.8491,
                    "f1_score": 0.6289,
                    "recall": 0.7861,
                }
            },
        }


@app.on_event("startup")
async def startup_event():
    load_model()


# ── Helper Functions ──────────────────────────────────────────────────────────

def customer_to_dataframe(customer: CustomerFeatures) -> pd.DataFrame:
    """Convert a CustomerFeatures pydantic model to a single-row DataFrame."""
    return pd.DataFrame([customer.model_dump()])


def run_prediction(df: pd.DataFrame) -> Dict[str, Any]:
    """Run pipeline prediction and return structured results."""
    proba = _pipeline.predict_proba(df)[:, 1]
    predictions = (proba >= DECISION_THRESHOLD).astype(int)

    results = []
    for p, pred in zip(proba, predictions):
        risk_label = "High Risk" if pred == 1 else "Low Risk"
        results.append({
            "churn_probability": round(float(p), 4),
            "churn_prediction": int(pred),
            "risk_label": risk_label,
        })
    return results


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """System health and model readiness check."""
    return HealthResponse(
        status="healthy",
        model_loaded=_pipeline is not None,
        startup_time=_startup_time,
        predictions_served=_predict_count,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """Return model metadata, performance metrics, and input schema."""
    if not _model_info:
        raise HTTPException(status_code=503, detail="Model metadata not available")
    perf = _model_info.get("performance", {}).get("holdout_test_set", {})
    return ModelInfoResponse(
        model_name=_model_info.get("model_name", "ChurnOps Champion"),
        version=_model_info.get("version", "latest"),
        algorithm=_model_info.get("algorithm", "XGBClassifier"),
        test_roc_auc=perf.get("roc_auc", 0.8491),
        test_f1=perf.get("f1_score", 0.6289),
        test_recall=perf.get("recall", 0.7861),
        decision_threshold=DECISION_THRESHOLD,
        trained_on=_model_info.get("training_data", {}).get("rows", 5634),
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict_single(customer: CustomerFeatures):
    """
    Predict churn probability for a single customer.

    Returns:
    - **churn_probability**: Float [0–1], probability of churn
    - **churn_prediction**: 0 (retained) or 1 (churned)
    - **risk_label**: "High Risk" or "Low Risk"
    """
    global _predict_count
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.perf_counter()
    try:
        df = customer_to_dataframe(customer)
        results = run_prediction(df)
        result = results[0]
        _predict_count += 1
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        return PredictResponse(
            customer_id=customer.customer_id or "UNKNOWN",
            churn_probability=result["churn_probability"],
            churn_prediction=result["churn_prediction"],
            risk_label=result["risk_label"],
            decision_threshold=DECISION_THRESHOLD,
            latency_ms=latency_ms,
            model_version=_model_info.get("version", "latest"),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictRequest):
    """
    Predict churn probability for a batch of customers (up to 1000).

    Returns one prediction per customer in the input list.
    """
    global _predict_count
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(request.customers) > 1000:
        raise HTTPException(status_code=400, detail="Batch size must be <= 1000")

    t0 = time.perf_counter()
    try:
        rows = [c.model_dump() for c in request.customers]
        df = pd.DataFrame(rows)
        results = run_prediction(df)
        _predict_count += len(results)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        predictions = []
        for customer, result in zip(request.customers, results):
            predictions.append(
                PredictResponse(
                    customer_id=customer.customer_id or f"IDX_{len(predictions)}",
                    churn_probability=result["churn_probability"],
                    churn_prediction=result["churn_prediction"],
                    risk_label=result["risk_label"],
                    decision_threshold=DECISION_THRESHOLD,
                    latency_ms=latency_ms / len(results),
                    model_version=_model_info.get("version", "latest"),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            )

        high_risk = sum(1 for p in predictions if p.churn_prediction == 1)
        return BatchPredictResponse(
            total=len(predictions),
            high_risk_count=high_risk,
            low_risk_count=len(predictions) - high_risk,
            predictions=predictions,
            latency_ms=latency_ms,
            model_version=_model_info.get("version", "latest"),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Batch prediction failed: {str(e)}")


@app.get("/", tags=["System"])
async def root():
    """API root — redirect hint to /docs."""
    return {
        "service": "ChurnOps Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
        "batch_predict": "/predict/batch",
    }
