"""
FastAPI application for heart disease prediction API.
"""

# flake8: noqa: F401, E501
import logging
import os
import sys
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.predict_model import HeartDiseasePredictor
from src.utils import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load configuration
config = load_config()

# Initialize model predictor
model_predictor = HeartDiseasePredictor(model_dir="models")

# Create FastAPI app
app = FastAPI(
    title="Heart Disease Prediction API",
    description="API for predicting heart disease risk based on clinical parameters",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# Define request models
class PatientData(BaseModel):
    """Patient clinical data for heart disease prediction."""

    age: int = Field(..., description="Age in years", example=61)
    sex: int = Field(..., description="Gender (0=female, 1=male)", example=1)
    cp: int = Field(..., description="Chest pain type (1-4)", example=3)
    trestbps: int = Field(..., description="Resting blood pressure (mm Hg)", example=140)
    chol: int = Field(..., description="Serum cholesterol (mg/dl)", example=240)
    fbs: int = Field(
        ..., description="Fasting blood sugar > 120 mg/dl (0=false, 1=true)", example=1
    )
    restecg: int = Field(..., description="Resting ECG results (0-2)", example=1)
    thalach: int = Field(..., description="Maximum heart rate achieved", example=150)
    exang: int = Field(..., description="Exercise induced angina (0=no, 1=yes)", example=1)
    oldpeak: float = Field(..., description="ST depression induced by exercise", example=2.4)
    slope: int = Field(..., description="Slope of the peak exercise ST segment (1-3)", example=2)
    ca: int = Field(
        ...,
        description="Number of major vessels colored by fluoroscopy (0-3)",
        example=1,
    )
    thal: int = Field(
        ...,
        description="Thalassemia (3=normal, 6=fixed defect, 7=reversible defect)",
        example=3,
    )

    class Config:
        schema_extra = {
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


class PredictionResponse(BaseModel):
    """Response model for heart disease prediction."""

    prediction: int = Field(..., description="Heart disease prediction (0=no, 1=yes)")
    probability: float = Field(..., description="Probability of heart disease")
    risk_level: str = Field(..., description="Risk level assessment")
    interpretation: Optional[str] = Field(None, description="Clinical interpretation of prediction")


# Define API endpoints
@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Heart Disease Prediction API. Use /docs for documentation."}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(patient_data: PatientData):
    """
    Predict heart disease risk from patient data.

    Args:
        patient_data: Patient clinical parameters

    Returns:
        Prediction result with probability and interpretation
    """
    logger.info("Received prediction request")

    try:
        # Convert patient data to dictionary
        patient_dict = patient_data.dict()

        # Make prediction
        prediction_result = model_predictor.predict(
            patient_dict, return_probabilities=True, return_interpretation=True
        )

        # Get ensemble prediction if available, otherwise use available model
        if "ensemble_predictions" in prediction_result:
            prediction = prediction_result["ensemble_predictions"][0]
            probability = prediction_result["ensemble_probabilities"][0]
        elif "sklearn_predictions" in prediction_result:
            prediction = prediction_result["sklearn_predictions"][0]
            probability = prediction_result["sklearn_probabilities"][0]
        elif "keras_predictions" in prediction_result:
            prediction = prediction_result["keras_predictions"][0]
            probability = prediction_result["keras_probabilities"][0]
        else:
            raise HTTPException(status_code=500, detail="No predictions available")

        # Determine risk level
        if probability < 0.3:
            risk_level = "LOW"
        elif probability < 0.6:
            risk_level = "MODERATE"
        else:
            risk_level = "HIGH"

        # Get interpretation if available
        interpretation = prediction_result.get("interpretation", None)

        # Construct response
        response = {
            "prediction": int(prediction),
            "probability": float(probability),
            "risk_level": risk_level,
            "interpretation": interpretation,
        }

        logger.info(f"Prediction: {prediction}, Probability: {probability:.4f}")

        return response

    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/models/info")
async def get_model_info():
    """
    Get information about the loaded models.

    Returns:
        Model information
    """
    logger.info("Received model info request")

    try:
        # Check available models
        models_available = []

        if model_predictor.has_sklearn_model:
            models_available.append("scikit-learn MLP")

        if model_predictor.has_keras_model:
            models_available.append("Keras MLP")

        # Construct response
        response = {
            "models_available": models_available,
            "ensemble_available": model_predictor.has_ensemble_model,
            "preprocessor_available": model_predictor.preprocessor is not None,
        }

        return response

    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")


if __name__ == "__main__":
    # Get API configuration from config
    host = config["api"]["host"]
    port = config["api"]["port"]

    # Start API server
    import uvicorn  # Import here to avoid flake8 E402 warning

    logger.info(f"Starting Heart Disease Prediction API on {host}:{port}")
    uvicorn.run("app:app", host=host, port=port, reload=True)
