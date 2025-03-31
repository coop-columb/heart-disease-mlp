"""
FastAPI application for heart disease prediction API.
This module provides REST API endpoints for predicting heart disease risk.
"""

import asyncio

# flake8: noqa: F401, E501
import logging
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Union

# Clean Python path to avoid conflicts with other projects
original_path = list(sys.path)

# Filter out any paths related to conflicting projects
filtered_path = [p for p in original_path if "EmotionAdaptiveMusic" not in p]
sys.path = filtered_path

# Add the project root first to ensure it takes precedence
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root in sys.path:
    sys.path.remove(project_root)
sys.path.insert(0, project_root)

# Configure logging before imports to ensure proper logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(project_root, "api.log"))],
)
logger = logging.getLogger(__name__)

# Log the Python path for debugging
logger.debug("Python path after cleaning:")
for p in sys.path:
    logger.debug(f"  {p}")

try:
    # Import FastAPI after path setup
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field

    # Import project modules
    from src.models.predict_model import HeartDiseasePredictor
    from src.utils import load_config

    logger.info("All modules imported successfully")
except ImportError as e:
    logger.critical(f"Error importing modules: {e}")
    traceback.print_exc()
    raise

# Load configuration
config = load_config()

# Batch processing configuration
BATCH_SIZE = config.get("api", {}).get("batch_size", 50)  # Default chunk size of 50 patients
MAX_WORKERS = config.get("api", {}).get("max_workers", 4)  # Default to 4 parallel workers
PERFORMANCE_LOGGING = config.get("api", {}).get("performance_logging", True)

# Initialize model predictor
model_predictor = HeartDiseasePredictor(model_dir="models")

# Initialize thread pool for parallel processing
thread_pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)

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

# Check if static directory exists, create if not
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=static_dir), name="static")


def process_patient_chunk(patients_chunk, model_name=None):
    """
    Process a chunk of patients in parallel.

    Args:
        patients_chunk: List of patient dictionaries to process
        model_name: Optional model to use for predictions

    Returns:
        List of prediction results for each patient
    """
    chunk_results = []

    for i, patient_dict in enumerate(patients_chunk):
        try:
            # Make prediction for this patient
            prediction_result = model_predictor.predict(
                patient_dict, return_probabilities=True, return_interpretation=True
            )

            # Check for error in prediction result
            if "error" in prediction_result:
                chunk_results.append(
                    {
                        "prediction": 0,
                        "probability": 0.0,
                        "risk_level": "ERROR",
                        "interpretation": f"Error: {prediction_result['error']}",
                        "model_used": None,
                    }
                )
                continue

            # Select model based on parameter
            model_used = None
            prediction = None
            probability = None

            # Try the specifically requested model first
            if model_name == "sklearn" and "sklearn_predictions" in prediction_result:
                prediction = prediction_result["sklearn_predictions"][0]
                probability = prediction_result.get("sklearn_probabilities", [None])[0]
                model_used = "sklearn_mlp"
            elif model_name == "keras" and "keras_predictions" in prediction_result:
                prediction = prediction_result["keras_predictions"][0]
                probability = prediction_result.get("keras_probabilities", [None])[0]
                model_used = "keras_mlp"
            elif model_name == "ensemble" and "ensemble_predictions" in prediction_result:
                prediction = prediction_result["ensemble_predictions"][0]
                probability = prediction_result.get("ensemble_probabilities", [None])[0]
                model_used = "ensemble"
            # If no specific model requested or requested model not available, use best available
            elif "ensemble_predictions" in prediction_result:
                prediction = prediction_result["ensemble_predictions"][0]
                probability = prediction_result.get("ensemble_probabilities", [None])[0]
                model_used = "ensemble"
            elif "sklearn_predictions" in prediction_result:
                prediction = prediction_result["sklearn_predictions"][0]
                probability = prediction_result.get("sklearn_probabilities", [None])[0]
                model_used = "sklearn_mlp"
            elif "keras_predictions" in prediction_result:
                prediction = prediction_result["keras_predictions"][0]
                probability = prediction_result.get("keras_probabilities", [None])[0]
                model_used = "keras_mlp"

            # Handle case where no predictions are available
            if prediction is None or model_used is None:
                chunk_results.append(
                    {
                        "prediction": 0,
                        "probability": 0.0,
                        "risk_level": "ERROR",
                        "interpretation": "No predictions available from any model",
                        "model_used": None,
                    }
                )
                continue

            # Handle missing probability
            if probability is None:
                probability = 0.5

            # Determine risk level
            if probability < 0.3:
                risk_level = "LOW"
            elif probability < 0.6:
                risk_level = "MODERATE"
            else:
                risk_level = "HIGH"

            # Get interpretation if available
            interpretation = prediction_result.get("interpretation", None)

            # Add result for this patient
            chunk_results.append(
                {
                    "prediction": int(prediction),
                    "probability": float(probability),
                    "risk_level": risk_level,
                    "interpretation": interpretation,
                    "model_used": model_used,
                }
            )

        except Exception as e:
            # Log and continue with next patient if one fails
            logger.error(f"Error processing patient: {str(e)}")
            chunk_results.append(
                {
                    "prediction": 0,
                    "probability": 0.0,
                    "risk_level": "ERROR",
                    "interpretation": f"Error processing patient: {str(e)}",
                    "model_used": None,
                }
            )

    return chunk_results


async def process_batch_optimized(patients_data, model_name=None):
    """
    Process a batch of patients using chunking and parallel processing.

    Args:
        patients_data: List of PatientData objects
        model_name: Optional model to use for predictions

    Returns:
        Tuple of (predictions, performance_metrics)
    """
    start_time = time.time()

    # Convert patients data to dictionaries
    patient_dicts = []
    for patient in patients_data:
        # Support both Pydantic v1 (dict) and v2 (model_dump)
        patient_dict = patient.model_dump() if hasattr(patient, "model_dump") else patient.dict()
        patient_dicts.append(patient_dict)

    # Split patients into chunks
    chunks = [patient_dicts[i : i + BATCH_SIZE] for i in range(0, len(patient_dicts), BATCH_SIZE)]
    logger.info(
        f"Processing {len(patient_dicts)} patients in {len(chunks)} chunks of size {BATCH_SIZE}"
    )

    # Process chunks in parallel using ThreadPoolExecutor
    tasks = []
    loop = asyncio.get_event_loop()
    for chunk in chunks:
        task = loop.run_in_executor(thread_pool, process_patient_chunk, chunk, model_name)
        tasks.append(task)

    # Wait for all chunks to complete
    chunk_results = await asyncio.gather(*tasks)

    # Flatten results
    all_results = []
    for chunk in chunk_results:
        all_results.extend(chunk)

    # Calculate performance metrics
    end_time = time.time()
    elapsed_time = end_time - start_time
    throughput = len(patient_dicts) / elapsed_time if elapsed_time > 0 else 0

    performance_metrics = {
        "total_patients": len(patient_dicts),
        "processing_time_seconds": round(elapsed_time, 3),
        "throughput_patients_per_second": round(throughput, 3),
        "num_chunks": len(chunks),
        "chunk_size": BATCH_SIZE,
        "num_workers": MAX_WORKERS,
    }

    logger.info(
        f"Batch processing completed in {elapsed_time:.2f} seconds. "
        f"Throughput: {throughput:.2f} patients/second"
    )

    return all_results, performance_metrics


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
        json_schema_extra = {
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


# Define API endpoints
@app.get("/")
async def root():
    """Root endpoint - serves the frontend UI."""
    return FileResponse(os.path.join(static_dir, "index.html"))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(patient_data: PatientData, model: str = None):
    """
    Predict heart disease risk from patient data.

    Args:
        patient_data: Patient clinical parameters
        model: Optional model to use (sklearn, keras, ensemble)

    Returns:
        Prediction result with probability and interpretation
    """
    logger.info(f"Received prediction request using model: {model}")

    try:
        # Convert patient data to dictionary
        # Support both Pydantic v1 (dict) and v2 (model_dump)
        patient_dict = (
            patient_data.model_dump()
            if hasattr(patient_data, "model_dump")
            else patient_data.dict()
        )

        # Make prediction
        prediction_result = model_predictor.predict(
            patient_dict, return_probabilities=True, return_interpretation=True
        )

        # Check for error in prediction result
        if "error" in prediction_result:
            logger.error(f"Prediction error: {prediction_result['error']}")
            raise HTTPException(status_code=500, detail=prediction_result["error"])

        # Select model based on parameter if provided
        model_used = None
        prediction = None
        probability = None

        # Try the specifically requested model first
        if model == "sklearn" and "sklearn_predictions" in prediction_result:
            prediction = prediction_result["sklearn_predictions"][0]
            probability = prediction_result.get("sklearn_probabilities", [None])[0]
            model_used = "sklearn_mlp"
        elif model == "keras" and "keras_predictions" in prediction_result:
            prediction = prediction_result["keras_predictions"][0]
            probability = prediction_result.get("keras_probabilities", [None])[0]
            model_used = "keras_mlp"
        elif model == "ensemble" and "ensemble_predictions" in prediction_result:
            prediction = prediction_result["ensemble_predictions"][0]
            probability = prediction_result.get("ensemble_probabilities", [None])[0]
            model_used = "ensemble"
        # If no specific model requested or requested model not available, use best available
        elif "ensemble_predictions" in prediction_result:
            prediction = prediction_result["ensemble_predictions"][0]
            probability = prediction_result.get("ensemble_probabilities", [None])[0]
            model_used = "ensemble"
        elif "sklearn_predictions" in prediction_result:
            prediction = prediction_result["sklearn_predictions"][0]
            probability = prediction_result.get("sklearn_probabilities", [None])[0]
            model_used = "sklearn_mlp"
        elif "keras_predictions" in prediction_result:
            prediction = prediction_result["keras_predictions"][0]
            probability = prediction_result.get("keras_probabilities", [None])[0]
            model_used = "keras_mlp"

        # Handle case where no predictions are available
        if prediction is None or model_used is None:
            logger.error("No valid predictions found in model output")
            raise HTTPException(status_code=500, detail="No predictions available from any model")

        # Handle missing probability
        if probability is None:
            logger.warning("Probability value missing, using default of 0.5")
            probability = 0.5

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
            "model_used": model_used,
        }

        logger.info(
            f"Prediction: {prediction}, Probability: {probability:.4f}, Model: {model_used}"
        )

        return response

    except HTTPException:
        # Re-raise HTTP exceptions without modification
        raise
    except ValueError as e:
        # Handle validation errors
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=422, detail=f"Invalid input data: {str(e)}")
    except Exception as e:
        # Handle all other errors
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(patients_data: List[PatientData], model: str = None):
    """
    Predict heart disease risk for multiple patients using optimized batch processing.

    Args:
        patients_data: List of patient clinical parameters
        model: Optional model to use (sklearn, keras, ensemble)

    Returns:
        Batch prediction result with probabilities and interpretations
    """
    logger.info(
        f"Received batch prediction request for {len(patients_data)} patients using model: {model}"
    )

    try:
        # If batch is empty, return an error
        if not patients_data:
            raise ValueError("No valid predictions could be made for any patients in the batch")

        # Process batch with optimized chunking and parallel processing
        results, performance_metrics = await process_batch_optimized(patients_data, model)

        # If no valid results were obtained, raise an exception
        if not results:
            raise ValueError("No valid predictions could be made for any patients in the batch")

        # Return batch results with performance metrics if enabled
        response = {"predictions": results}
        if PERFORMANCE_LOGGING:
            response["performance_metrics"] = performance_metrics

        return response

    except ValueError as e:
        # Handle validation errors
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=422, detail=f"Invalid input data: {str(e)}")
    except Exception as e:
        # Handle all other errors
        logger.error(f"Error making batch predictions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


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
        models_available = {}

        if model_predictor.has_sklearn_model:
            models_available["sklearn_mlp"] = True

        if model_predictor.has_keras_model:
            models_available["keras_mlp"] = True

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


class BatchConfig(BaseModel):
    """Configuration for batch processing."""

    batch_size: Optional[int] = Field(
        None, description="Number of patients to process in each chunk", ge=1, le=1000
    )
    max_workers: Optional[int] = Field(
        None, description="Maximum number of worker threads for parallel processing", ge=1, le=20
    )
    performance_logging: Optional[bool] = Field(
        None, description="Whether to include performance metrics in the response"
    )


@app.get("/batch/config")
async def get_batch_config():
    """
    Get current batch processing configuration.

    Returns:
        Current batch processing configuration
    """
    logger.info("Received batch configuration request")

    return {
        "batch_size": BATCH_SIZE,
        "max_workers": MAX_WORKERS,
        "performance_logging": PERFORMANCE_LOGGING,
    }


@app.post("/batch/config")
async def update_batch_config(config: BatchConfig):
    """
    Update batch processing configuration.

    Args:
        config: New batch processing configuration

    Returns:
        Updated batch processing configuration
    """
    global BATCH_SIZE, MAX_WORKERS, PERFORMANCE_LOGGING, thread_pool

    logger.info(f"Received batch configuration update request: {config}")

    try:
        # Update configuration parameters if provided
        if config.batch_size is not None:
            BATCH_SIZE = config.batch_size

        if config.max_workers is not None and config.max_workers != MAX_WORKERS:
            MAX_WORKERS = config.max_workers
            # Recreate thread pool with new worker count
            thread_pool.shutdown(wait=True)
            thread_pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)

        if config.performance_logging is not None:
            PERFORMANCE_LOGGING = config.performance_logging

        logger.info(
            f"Batch configuration updated: batch_size={BATCH_SIZE}, "
            f"max_workers={MAX_WORKERS}, performance_logging={PERFORMANCE_LOGGING}"
        )

        return {
            "batch_size": BATCH_SIZE,
            "max_workers": MAX_WORKERS,
            "performance_logging": PERFORMANCE_LOGGING,
        }

    except Exception as e:
        logger.error(f"Error updating batch configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating batch configuration: {str(e)}")


if __name__ == "__main__":
    # Get API configuration from config
    host = config["api"]["host"]
    port = config["api"]["port"]

    # Start API server
    import uvicorn  # Import here to avoid flake8 E402 warning

    logger.info(f"Starting Heart Disease Prediction API on {host}:{port}")
    uvicorn.run("app:app", host=host, port=port, reload=True)
