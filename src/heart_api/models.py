from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PatientData(BaseModel):
    """Patient clinical data for heart disease prediction."""

    age: int = Field(..., description="Age in years")
    sex: int = Field(..., description="Gender (0=female, 1=male)")
    cp: int = Field(..., description="Chest pain type (1-4)")
    trestbps: int = Field(..., description="Resting blood pressure (mm Hg)")
    chol: int = Field(..., description="Serum cholesterol (mg/dl)")
    fbs: int = Field(
        ..., description="Fasting blood sugar > 120 mg/dl (0=false, 1=true)"
    )
    restecg: int = Field(..., description="Resting ECG results (0-2)")
    thalach: int = Field(..., description="Maximum heart rate achieved")
    exang: int = Field(..., description="Exercise induced angina (0=no, 1=yes)")
    oldpeak: float = Field(..., description="ST depression induced by exercise")
    slope: int = Field(..., description="Slope of the peak exercise ST segment (1-3)")
    ca: int = Field(
        ..., description="Number of major vessels colored by fluoroscopy (0-3)"
    )
    thal: int = Field(
        ..., description="Thalassemia (3=normal, 6=fixed defect, 7=reversible defect)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 61,
                "sex": 1,
                "cp": 3,
                "trestbps": 140,
                "chol": 240,
                "fbs": 1,
                "restecg": 1,
                "thalach": 150,
                "exang": 1,
                "oldpeak": 2.4,
                "slope": 2,
                "ca": 1,
                "thal": 3,
            }
        }
    }


class PredictionResponse(BaseModel):
    """Response model for heart disease prediction."""

    prediction: int = Field(
        ..., description="Heart disease prediction (0=no, 1=yes, 0 if error)"
    )
    probability: float = Field(
        ..., description="Probability of heart disease (0.0 if error)"
    )
    risk_level: str = Field(
        ..., description="Risk level assessment (LOW, MODERATE, HIGH, or ERROR)"
    )
    interpretation: Optional[str] = Field(
        None, description="Clinical interpretation of prediction or error message"
    )
    model_used: Optional[str] = Field(
        None, description="Model used for prediction (None if error)"
    )


class BatchPredictionResponse(BaseModel):
    """Response model for batch heart disease prediction."""

    predictions: List[PredictionResponse] = Field(
        ...,
        description="List of predictions for each patient, including any that resulted in errors",
    )
    performance_metrics: Optional[Dict[str, float]] = Field(
        None,
        description="Performance metrics for the batch prediction operation (if enabled)",
    )


class BatchConfig(BaseModel):
    """Configuration for batch processing."""

    batch_size: Optional[int] = Field(
        None, description="Number of patients to process in each chunk", ge=1, le=1000
    )
    max_workers: Optional[int] = Field(
        None,
        description="Maximum number of worker threads for parallel processing",
        ge=1,
        le=20,
    )
    performance_logging: Optional[bool] = Field(
        None, description="Whether to include performance metrics in the response"
    )


class CacheConfig(BaseModel):
    """Configuration for prediction caching."""

    enabled: Optional[bool] = Field(
        None, description="Whether prediction caching is enabled"
    )
    max_size: Optional[int] = Field(
        None, description="Maximum number of cached entries", ge=1, le=10000
    )
    ttl: Optional[int] = Field(
        None, description="Time-to-live in seconds for cache entries", ge=1, le=86400
    )
