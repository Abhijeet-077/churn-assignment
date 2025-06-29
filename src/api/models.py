"""
Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from enum import Enum


class GenderEnum(str, Enum):
    """Gender enumeration."""
    male = "Male"
    female = "Female"


class YesNoEnum(str, Enum):
    """Yes/No enumeration."""
    yes = "Yes"
    no = "No"


class InternetServiceEnum(str, Enum):
    """Internet service enumeration."""
    dsl = "DSL"
    fiber_optic = "Fiber optic"
    no = "No"


class ContractEnum(str, Enum):
    """Contract type enumeration."""
    month_to_month = "Month-to-month"
    one_year = "One year"
    two_year = "Two year"


class PaymentMethodEnum(str, Enum):
    """Payment method enumeration."""
    electronic_check = "Electronic check"
    mailed_check = "Mailed check"
    bank_transfer = "Bank transfer (automatic)"
    credit_card = "Credit card (automatic)"


class CustomerData(BaseModel):
    """Customer data model for churn prediction."""
    
    # Demographics
    gender: GenderEnum = Field(..., description="Customer gender")
    senior_citizen: int = Field(..., ge=0, le=1, description="Whether customer is senior citizen (0 or 1)")
    partner: YesNoEnum = Field(..., description="Whether customer has partner")
    dependents: YesNoEnum = Field(..., description="Whether customer has dependents")
    
    # Account information
    tenure: int = Field(..., ge=0, le=100, description="Number of months customer has stayed")
    
    # Services
    phone_service: YesNoEnum = Field(..., description="Whether customer has phone service")
    multiple_lines: str = Field(..., description="Whether customer has multiple lines")
    internet_service: InternetServiceEnum = Field(..., description="Internet service provider")
    online_security: str = Field(..., description="Whether customer has online security")
    online_backup: str = Field(..., description="Whether customer has online backup")
    device_protection: str = Field(..., description="Whether customer has device protection")
    tech_support: str = Field(..., description="Whether customer has tech support")
    streaming_tv: str = Field(..., description="Whether customer has streaming TV")
    streaming_movies: str = Field(..., description="Whether customer has streaming movies")
    
    # Contract and billing
    contract: ContractEnum = Field(..., description="Contract term")
    paperless_billing: YesNoEnum = Field(..., description="Whether customer has paperless billing")
    payment_method: PaymentMethodEnum = Field(..., description="Payment method")
    monthly_charges: float = Field(..., ge=0, le=200, description="Monthly charges amount")
    total_charges: float = Field(..., ge=0, description="Total charges amount")
    
    @validator('multiple_lines', 'online_security', 'online_backup', 'device_protection', 
              'tech_support', 'streaming_tv', 'streaming_movies')
    def validate_service_fields(cls, v):
        """Validate service fields."""
        valid_values = ["Yes", "No", "No internet service", "No phone service"]
        if v not in valid_values:
            raise ValueError(f"Value must be one of: {valid_values}")
        return v
    
    @validator('total_charges')
    def validate_total_charges(cls, v, values):
        """Validate total charges against monthly charges and tenure."""
        if 'monthly_charges' in values and 'tenure' in values:
            monthly = values['monthly_charges']
            tenure = values['tenure']
            if v > 0 and tenure > 0:
                # Allow some flexibility in total charges calculation
                expected_min = monthly * tenure * 0.5  # 50% of expected
                expected_max = monthly * tenure * 2.0  # 200% of expected
                if not (expected_min <= v <= expected_max):
                    raise ValueError(
                        f"Total charges {v} seems inconsistent with monthly charges {monthly} "
                        f"and tenure {tenure}. Expected range: {expected_min:.2f} - {expected_max:.2f}"
                    )
        return v


class BatchCustomerData(BaseModel):
    """Batch prediction request model."""
    customers: List[CustomerData] = Field(..., min_items=1, max_items=1000, 
                                        description="List of customers for batch prediction")


class PredictionResponse(BaseModel):
    """Single prediction response model."""
    customer_id: Optional[str] = Field(None, description="Customer identifier")
    churn_probability: float = Field(..., ge=0, le=1, description="Probability of churn (0-1)")
    churn_prediction: str = Field(..., description="Predicted churn class (Yes/No)")
    confidence: str = Field(..., description="Prediction confidence level")
    risk_level: str = Field(..., description="Customer risk level")
    model_used: str = Field(..., description="Model used for prediction")
    
    
class BatchPredictionResponse(BaseModel):
    """Batch prediction response model."""
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    summary: Dict[str, Any] = Field(..., description="Batch prediction summary")


class ModelInfo(BaseModel):
    """Model information response."""
    model_name: str = Field(..., description="Name of the model")
    model_type: str = Field(..., description="Type of the model")
    accuracy: float = Field(..., description="Model accuracy")
    roc_auc: float = Field(..., description="Model ROC-AUC score")
    precision: float = Field(..., description="Model precision")
    recall: float = Field(..., description="Model recall")
    f1_score: float = Field(..., description="Model F1 score")
    training_date: str = Field(..., description="Model training date")
    feature_count: int = Field(..., description="Number of features used")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")
    models_loaded: List[str] = Field(..., description="List of loaded models")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")


class FeatureImportanceResponse(BaseModel):
    """Feature importance response model."""
    model_name: str = Field(..., description="Name of the model")
    feature_importance: Dict[str, float] = Field(..., description="Feature importance scores")
    top_features: List[str] = Field(..., description="Top 10 most important features")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: str = Field(..., description="Error timestamp")


class PredictionRequest(BaseModel):
    """Single prediction request model."""
    customer_data: CustomerData = Field(..., description="Customer data for prediction")
    model_name: Optional[str] = Field("best", description="Model to use for prediction")
    include_explanation: bool = Field(False, description="Include prediction explanation")


class ExplanationResponse(BaseModel):
    """Prediction explanation response model."""
    feature_contributions: Dict[str, float] = Field(..., description="Feature contributions to prediction")
    top_positive_factors: List[str] = Field(..., description="Top factors increasing churn probability")
    top_negative_factors: List[str] = Field(..., description="Top factors decreasing churn probability")
    explanation_text: str = Field(..., description="Human-readable explanation")


class DetailedPredictionResponse(PredictionResponse):
    """Detailed prediction response with explanation."""
    explanation: Optional[ExplanationResponse] = Field(None, description="Prediction explanation")
    feature_values: Dict[str, Any] = Field(..., description="Input feature values")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
