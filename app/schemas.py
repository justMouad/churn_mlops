from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field

Contract = Literal["month-to-month", "one-year", "two-year"]
Payment = Literal["credit_card", "bank_transfer", "electronic_check", "mailed_check"]


class Customer(BaseModel):
    customer_id: Optional[str] = None
    tenure_months: int = Field(ge=0, le=120)
    monthly_charges: float = Field(gt=0, le=1000)
    contract_type: Contract
    payment_method: Payment
    support_tickets_90d: int = Field(ge=0, le=100)
    usage_hours_30d: float = Field(ge=0, le=1000)
    late_payments_6m: int = Field(ge=0, le=50)
    num_products: int = Field(ge=1, le=20)
    has_discount: int = Field(ge=0, le=1)
    age: int = Field(ge=18, le=110)

    model_config = {"json_schema_extra": {"example": {
        "customer_id": "C-1001", "tenure_months": 5, "monthly_charges": 92.5, "contract_type": "month-to-month",
        "payment_method": "electronic_check", "support_tickets_90d": 4, "usage_hours_30d": 22.0,
        "late_payments_6m": 2, "num_products": 1, "has_discount": 0, "age": 34}}}


class ChurnPrediction(BaseModel):
    customer_id: Optional[str]
    churn_probability: float
    churn_predicted: bool
    risk_tier: Literal["low", "medium", "high"]
    threshold: float
    model_version: str


class BatchRequest(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    predictions: list[ChurnPrediction]


class ForecastRequest(BaseModel):
    horizon: int = Field(default=28, ge=1, le=180)
    model: Literal["champion", "prophet", "lstm", "naive"] = "champion"


class ForecastPoint(BaseModel):
    ds: date
    yhat: float
    yhat_lower: Optional[float] = None
    yhat_upper: Optional[float] = None


class ForecastResponse(BaseModel):
    model: str
    horizon: int
    last_observation: date
    forecast: list[ForecastPoint]


class Actual(BaseModel):
    ds: date
    y: float = Field(ge=0)


class ForecastPerfRequest(BaseModel):
    actuals: list[Actual] = Field(min_length=7, max_length=365)
