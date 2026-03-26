"""
schemas.py — Pydantic input/output models for Customer Churn prediction.

Defines validated data models matching the Telco Customer Churn dataset features.
"""
from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional


class CustomerInput(BaseModel):
    """Validated input schema for churn prediction requests.

    Each field maps to a column in the Telco Customer Churn dataset.
    """

    gender: Literal["Male", "Female"] = Field(
        ..., description="Customer gender"
    )
    SeniorCitizen: int = Field(
        ..., ge=0, le=1, description="Whether the customer is a senior citizen (1=yes, 0=no)"
    )
    Partner: Literal["Yes", "No"] = Field(
        ..., description="Whether the customer has a partner"
    )
    Dependents: Literal["Yes", "No"] = Field(
        ..., description="Whether the customer has dependents"
    )
    tenure: int = Field(
        ..., ge=0, le=100, description="Number of months the customer has been with the company"
    )
    PhoneService: Literal["Yes", "No"] = Field(
        ..., description="Whether the customer has phone service"
    )
    MultipleLines: Literal["Yes", "No", "No phone service"] = Field(
        ..., description="Whether the customer has multiple lines"
    )
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(
        ..., description="Customer's internet service provider"
    )
    OnlineSecurity: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Whether the customer has online security"
    )
    OnlineBackup: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Whether the customer has online backup"
    )
    DeviceProtection: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Whether the customer has device protection"
    )
    TechSupport: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Whether the customer has tech support"
    )
    StreamingTV: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Whether the customer has streaming TV"
    )
    StreamingMovies: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Whether the customer has streaming movies"
    )
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ..., description="The contract term of the customer"
    )
    PaperlessBilling: Literal["Yes", "No"] = Field(
        ..., description="Whether the customer has paperless billing"
    )
    PaymentMethod: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ] = Field(..., description="The customer's payment method")
    MonthlyCharges: float = Field(
        ..., ge=0, le=500, description="Monthly charge amount"
    )
    TotalCharges: float = Field(
        ..., ge=0, description="Total charges to date"
    )

    @field_validator("TotalCharges")
    @classmethod
    def total_charges_must_be_plausible(cls, v: float, info) -> float:
        """Total charges should be at least monthly charges (for 1+ month tenure)."""
        tenure = info.data.get("tenure")
        monthly = info.data.get("MonthlyCharges")
        if tenure is not None and monthly is not None and tenure > 0:
            if v < monthly * 0.5:
                raise ValueError(
                    f"TotalCharges ({v}) seems too low for tenure={tenure} "
                    f"months with MonthlyCharges={monthly}"
                )
        return v


class ChurnPrediction(BaseModel):
    """Validated output schema returned after inference."""

    prediction: Literal["Churn", "No Churn"]
    churn_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Probability that the customer will churn"
    )
    model_version: str = Field(
        default="churn_gb_v1", description="Model version identifier"
    )
