"""
Phase 22 — FastAPI Schemas
============================
Pydantic v2 request/response models for the ChurnOps inference API.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class CustomerFeatures(BaseModel):
    """Raw customer features (19 fields) — matches the Telco CSV schema minus customerID."""

    customer_id: Optional[str] = Field(None, description="Optional customer identifier")

    # Demographics
    gender: str = Field("Male", description="Gender (Male/Female) — dropped internally")
    SeniorCitizen: int = Field(0, ge=0, le=1, description="Senior citizen flag (0 or 1)")
    Partner: str = Field("No", description="Has partner (Yes/No)")
    Dependents: str = Field("No", description="Has dependents (Yes/No)")

    # Account
    tenure: int = Field(12, ge=0, le=72, description="Months with company [0, 72]")
    Contract: str = Field("Month-to-month", description="Contract type")
    PaperlessBilling: str = Field("Yes", description="Paperless billing (Yes/No)")
    PaymentMethod: str = Field("Electronic check", description="Payment method")
    MonthlyCharges: float = Field(79.85, ge=0.0, description="Monthly charge ($)")
    TotalCharges: float = Field(0.0, ge=0.0, description="Total charges ($)")

    # Services
    PhoneService: str = Field("Yes", description="Phone service (Yes/No) — dropped internally")
    MultipleLines: str = Field("No", description="Multiple lines")
    InternetService: str = Field("Fiber optic", description="Internet service type")
    OnlineSecurity: str = Field("No", description="Online security service")
    OnlineBackup: str = Field("No", description="Online backup service")
    DeviceProtection: str = Field("No", description="Device protection service")
    TechSupport: str = Field("No", description="Tech support service")
    StreamingTV: str = Field("No", description="Streaming TV service")
    StreamingMovies: str = Field("No", description="Streaming movies service")

    model_config = {"extra": "ignore"}


class BatchPredictRequest(BaseModel):
    """Batch prediction request — list of customers."""
    customers: List[CustomerFeatures] = Field(..., min_length=1, max_length=1000)


class PredictResponse(BaseModel):
    """Single customer prediction response."""
    customer_id: str
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    churn_prediction: int = Field(..., ge=0, le=1)
    risk_label: str
    decision_threshold: float
    latency_ms: float
    model_version: str
    timestamp: str


class BatchPredictResponse(BaseModel):
    """Batch prediction response."""
    total: int
    high_risk_count: int
    low_risk_count: int
    predictions: List[PredictResponse]
    latency_ms: float
    model_version: str
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    startup_time: str
    predictions_served: int
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Model metadata response."""
    model_name: str
    version: str
    algorithm: str
    test_roc_auc: float
    test_f1: float
    test_recall: float
    decision_threshold: float
    trained_on: int
