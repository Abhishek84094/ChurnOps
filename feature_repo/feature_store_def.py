"""
Phase 19 — Feast Feature Store Configuration
=============================================
Defines the ChurnOps Feature Store using Feast 0.38+ with a local SQLite
online store and file-based offline store.

Features served:
  - Customer demographics (SeniorCitizen, Partner, Dependents)
  - Service usage (tenure, contract, internet service)
  - Financial signals (MonthlyCharges, TotalCharges)
  - Engineered features (is_month_to_month, has_fiber_optic, is_auto_pay, etc.)

Usage:
    feast apply                    # Register feature definitions
    feast materialize-incremental  # Sync offline → online store
    feast serve                    # Start feature server
"""

from datetime import timedelta

from feast import (
    Entity,
    FeatureService,
    FeatureView,
    Field,
    FileSource,
)
from feast.types import Float32, Int32, String

# ─── Data Source ───────────────────────────────────────────────────────────────
# Offline store backed by the processed training CSV
churn_source = FileSource(
    name="churn_offline_source",
    path="../data/interim/telco_engineered.csv",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
    description="IBM Telco Customer Churn — engineered feature set",
)

# ─── Entity ────────────────────────────────────────────────────────────────────
customer = Entity(
    name="customer_id",
    description="Unique telecom customer identifier",
    tags={"owner": "ChurnOps Team"},
)

# ─── Feature Views ─────────────────────────────────────────────────────────────

customer_demographics_fv = FeatureView(
    name="customer_demographics",
    entities=[customer],
    ttl=timedelta(days=90),
    schema=[
        Field(name="SeniorCitizen", dtype=Int32),
        Field(name="Partner", dtype=String),
        Field(name="Dependents", dtype=String),
        Field(name="gender", dtype=String),
    ],
    source=churn_source,
    tags={"team": "data-eng", "domain": "demographics"},
    description="Customer demographic features (SeniorCitizen, Partner, Dependents, Gender)",
)

customer_account_fv = FeatureView(
    name="customer_account",
    entities=[customer],
    ttl=timedelta(days=30),
    schema=[
        Field(name="tenure", dtype=Int32),
        Field(name="Contract", dtype=String),
        Field(name="PaperlessBilling", dtype=String),
        Field(name="PaymentMethod", dtype=String),
        Field(name="MonthlyCharges", dtype=Float32),
        Field(name="TotalCharges", dtype=Float32),
    ],
    source=churn_source,
    tags={"team": "data-eng", "domain": "account"},
    description="Customer account and billing features",
)

customer_services_fv = FeatureView(
    name="customer_services",
    entities=[customer],
    ttl=timedelta(days=30),
    schema=[
        Field(name="PhoneService", dtype=String),
        Field(name="MultipleLines", dtype=String),
        Field(name="InternetService", dtype=String),
        Field(name="OnlineSecurity", dtype=String),
        Field(name="OnlineBackup", dtype=String),
        Field(name="DeviceProtection", dtype=String),
        Field(name="TechSupport", dtype=String),
        Field(name="StreamingTV", dtype=String),
        Field(name="StreamingMovies", dtype=String),
    ],
    source=churn_source,
    tags={"team": "data-eng", "domain": "services"},
    description="Customer subscribed services",
)

customer_engineered_fv = FeatureView(
    name="customer_engineered_features",
    entities=[customer],
    ttl=timedelta(days=30),
    schema=[
        Field(name="is_month_to_month", dtype=Int32),
        Field(name="has_fiber_optic", dtype=Int32),
        Field(name="is_auto_pay", dtype=Int32),
        Field(name="has_family", dtype=Int32),
        Field(name="has_security_or_techsupport", dtype=Int32),
        Field(name="total_services", dtype=Int32),
        Field(name="tenure_cohort", dtype=String),
    ],
    source=churn_source,
    tags={"team": "ml-eng", "domain": "engineered", "phase": "5"},
    description="Domain-engineered features with proven churn predictive signal",
)

# ─── Feature Service ───────────────────────────────────────────────────────────

churn_prediction_service = FeatureService(
    name="churn_prediction_service_v1",
    features=[
        customer_demographics_fv,
        customer_account_fv,
        customer_services_fv,
        customer_engineered_fv,
    ],
    description=(
        "Feature service for ChurnOps v1 prediction API. "
        "Serves all 19 raw + 7 engineered features for the XGBoost champion model."
    ),
    tags={
        "model": "ChurnOps_Champion_XGBoost",
        "version": "1.0.0",
        "owner": "ChurnOps Team",
    },
)
