"""
ChurnOps — Feast Feature Definitions
Phase 19: Feature Store
Generated: 2026-10-06T17:06:58.269165+00:00
"""

from datetime import timedelta
from feast import Entity, FeatureService, FeatureView, Field, FileSource
from feast.types import Float32, Int64, String

# Entity
customer = Entity(
    name="customer_id",
    description="Unique customer identifier for telco churn prediction",
    join_keys=["customer_id"],
)

# Data Source
customer_raw_source = FileSource(
    path="data/processed/train_feast.parquet",
    timestamp_field="event_timestamp",
)

# Feature Views
demographic_fv = FeatureView(
    name="customer_demographics",
    entities=[customer],
    ttl=timedelta(days=90),
    schema=[
        Field(name="SeniorCitizen", dtype=Int64),
        Field(name="tenure", dtype=Int64),
        Field(name="Partner", dtype=String),
        Field(name="Dependents", dtype=String),
    ],
    source=customer_raw_source,
)

service_fv = FeatureView(
    name="customer_services",
    entities=[customer],
    ttl=timedelta(days=90),
    schema=[
        Field(name="InternetService", dtype=String),
        Field(name="OnlineSecurity", dtype=String),
        Field(name="TechSupport", dtype=String),
        Field(name="StreamingTV", dtype=String),
        Field(name="StreamingMovies", dtype=String),
    ],
    source=customer_raw_source,
)

billing_fv = FeatureView(
    name="customer_billing",
    entities=[customer],
    ttl=timedelta(days=30),
    schema=[
        Field(name="Contract", dtype=String),
        Field(name="PaymentMethod", dtype=String),
        Field(name="MonthlyCharges", dtype=Float32),
        Field(name="TotalCharges", dtype=Float32),
    ],
    source=customer_raw_source,
)

engineered_fv = FeatureView(
    name="customer_engineered",
    entities=[customer],
    ttl=timedelta(days=90),
    schema=[
        Field(name="tenure_cohort", dtype=String),
        Field(name="total_services", dtype=Int64),
        Field(name="is_auto_pay", dtype=Int64),
        Field(name="has_fiber_optic", dtype=Int64),
        Field(name="has_security_or_techsupport", dtype=Int64),
        Field(name="has_family", dtype=Int64),
        Field(name="is_month_to_month", dtype=Int64),
    ],
    source=customer_raw_source,
)

churn_prediction_service = FeatureService(
    name="churn_prediction_v1",
    features=[demographic_fv, service_fv, billing_fv, engineered_fv],
)