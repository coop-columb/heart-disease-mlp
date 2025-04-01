from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class PatientData(BaseModel):
    """Patient clinical data for heart disease prediction."""

    age: int = Field(
        ...,
        description="Age in years",
        ge=18,
        le=100,
        json_schema_extra={
            "error_messages": {
                "ge": "Age must be at least 18 years",
                "le": "Age must be at most 100 years",
            }
        },
    )
    sex: int = Field(
        ...,
        description="Gender (0=female, 1=male)",
        ge=0,
        le=1,
        json_schema_extra={
            "error_messages": {
                "ge": "Sex must be 0 (female) or 1 (male)",
                "le": "Sex must be 0 (female) or 1 (male)",
            }
        },
    )
    cp: int = Field(
        ...,
        description="Chest pain type (1-4)",
        ge=1,
        le=4,
        json_schema_extra={
            "error_messages": {
                "ge": "Chest pain type must be at least 1",
                "le": "Chest pain type must be at most 4",
            }
        },
    )
    trestbps: int = Field(
        ...,
        description="Resting blood pressure (mm Hg)",
        ge=60,
        le=300,
        json_schema_extra={
            "error_messages": {
                "ge": "Resting blood pressure must be at least 60 mm Hg",
                "le": "Resting blood pressure must be at most 300 mm Hg",
            }
        },
    )
    chol: int = Field(
        ...,
        description="Serum cholesterol (mg/dl)",
        ge=100,
        le=600,
        json_schema_extra={
            "error_messages": {
                "ge": "Serum cholesterol must be at least 100 mg/dl",
                "le": "Serum cholesterol must be at most 600 mg/dl",
            }
        },
    )
    fbs: int = Field(
        ...,
        description="Fasting blood sugar > 120 mg/dl (0=false, 1=true)",
        ge=0,
        le=1,
        json_schema_extra={
            "error_messages": {
                "ge": "Fasting blood sugar must be 0 (false) or 1 (true)",
                "le": "Fasting blood sugar must be 0 (false) or 1 (true)",
            }
        },
    )
    restecg: int = Field(
        ...,
        description="Resting ECG results (0-2)",
        ge=0,
        le=2,
        json_schema_extra={
            "error_messages": {
                "ge": "Resting ECG result must be between 0 and 2",
                "le": "Resting ECG result must be between 0 and 2",
            }
        },
    )
    thalach: int = Field(
        ...,
        description="Maximum heart rate achieved",
        ge=60,
        le=220,
        json_schema_extra={
            "error_messages": {
                "ge": "Maximum heart rate must be at least 60 bpm",
                "le": "Maximum heart rate must be at most 220 bpm",
            }
        },
    )
    exang: int = Field(
        ...,
        description="Exercise induced angina (0=no, 1=yes)",
        ge=0,
        le=1,
        json_schema_extra={
            "error_messages": {
                "ge": "Exercise induced angina must be 0 (no) or 1 (yes)",
                "le": "Exercise induced angina must be 0 (no) or 1 (yes)",
            }
        },
    )
    oldpeak: float = Field(
        ...,
        description="ST depression induced by exercise",
        ge=0.0,
        le=10.0,
        json_schema_extra={
            "error_messages": {
                "ge": "ST depression must be at least 0.0",
                "le": "ST depression must be at most 10.0",
            }
        },
    )
    slope: int = Field(
        ...,
        description="Slope of the peak exercise ST segment (1-3)",
        ge=1,
        le=3,
        json_schema_extra={
            "error_messages": {
                "ge": "Slope must be between 1 and 3",
                "le": "Slope must be between 1 and 3",
            }
        },
    )
    ca: int = Field(
        ...,
        description="Number of major vessels colored by fluoroscopy (0-3)",
        ge=0,
        le=3,
        json_schema_extra={
            "error_messages": {
                "ge": "Number of major vessels must be between 0 and 3",
                "le": "Number of major vessels must be between 0 and 3",
            }
        },
    )
    thal: int = Field(
        ...,
        description=("Thalassemia " "(3=normal, 6=fixed defect, 7=reversible defect)"),
    )

    @field_validator("thal")
    def validate_thal(cls, value):
        valid_values = [3, 6, 7]
        if value not in valid_values:
            raise ValueError(
                f"Thalassemia value must be one of {valid_values} "
                "(3=normal, 6=fixed defect, 7=reversible defect)"
            )
        return value

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

    prediction: int = Field(..., description="Heart disease prediction (0=no, 1=yes, 0 if error)")
    probability: float = Field(..., description="Probability of heart disease (0.0 if error)")
    risk_level: str = Field(
        ..., description="Risk level assessment (LOW, MODERATE, HIGH, or ERROR)"
    )
    interpretation: Optional[str] = Field(
        None, description="Clinical interpretation of prediction or error message"
    )
    model_used: Optional[str] = Field(None, description="Model used for prediction (None if error)")


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
        None,
        description="Number of patients to process in each chunk",
        ge=1,
        le=1000,
        json_schema_extra={
            "error_messages": {
                "ge": "Batch size must be at least 1",
                "le": "Batch size must be at most 1000",
            }
        },
    )
    max_workers: Optional[int] = Field(
        None,
        description="Maximum number of worker threads for parallel processing",
        ge=1,
        le=20,
        json_schema_extra={
            "error_messages": {
                "ge": "Maximum workers must be at least 1",
                "le": "Maximum workers must be at most 20",
            }
        },
    )
    performance_logging: Optional[bool] = Field(
        None, description="Whether to include performance metrics in the response"
    )


class CacheConfig(BaseModel):
    """Configuration for prediction caching."""

    enabled: Optional[bool] = Field(None, description="Whether prediction caching is enabled")
    max_size: Optional[int] = Field(
        None,
        description="Maximum number of cached entries",
        ge=1,
        le=10000,
        json_schema_extra={
            "error_messages": {
                "ge": "Maximum cache size must be at least 1",
                "le": "Maximum cache size must be at most 10000",
            }
        },
    )
    ttl: Optional[int] = Field(
        None,
        description="Time-to-live in seconds for cache entries",
        ge=1,
        le=86400,
        json_schema_extra={
            "error_messages": {
                "ge": "TTL must be at least 1 second",
                "le": "TTL must be at most 86400 seconds (24 hours)",
            }
        },
    )
