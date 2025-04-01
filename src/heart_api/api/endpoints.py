# Standard library imports
import asyncio
import logging
import os
import time
from datetime import datetime
from typing import List, Optional

# Third-party imports
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
from pydantic import BaseModel, Field

# Local application imports
from heart_api.core import (
    BATCH_SIZE,
    MAX_WORKERS,
    PERFORMANCE_LOGGING,
    auth_handler,
    auth_settings,
    model_predictor,
    thread_pool,
)
from heart_api.models import BatchPredictionResponse, PatientData, PredictionResponse

# Initialize logger and router
logger = logging.getLogger(__name__)
router = APIRouter()

# Store the server start time for uptime
SERVER_START_TIME = time.time()

# Setup authentication schemes
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token", auto_error=False)
api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


class AuthenticationError(HTTPException):
    def __init__(self, detail: str = "Invalid authentication credentials"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )


def validate_batch_request(data: List[dict]):
    if not data:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Batch request cannot be empty"
        )
    if len(data) > 100:  # example limit
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Batch size exceeds maximum limit of 100",
        )


async def verify_api_key(api_key: str = Depends(api_key_scheme)):
    """Verify API key."""
    if not auth_settings.enabled:
        return api_key
    if not api_key:
        raise AuthenticationError("API key is required")
    if not auth_handler.verify_authentication(api_key):
        raise AuthenticationError("Invalid API key")
    return api_key


async def verify_authentication(
    request: Request,
    token: Optional[str] = Depends(oauth2_scheme),
    api_key: Optional[str] = Depends(api_key_scheme),
):
    """Verify authentication for protected endpoints."""
    if not auth_settings.enabled:
        return True

    path = request.url.path
    if path in auth_settings.public_endpoints:
        return True

    # Check API key first
    if api_key and auth_handler.verify_authentication(api_key):
        return True

    # Then check bearer token
    if token and await auth_handler.verify_token(token):
        return True

    raise AuthenticationError("Could not validate credentials")


def get_risk_level(probability):
    """Determine risk level based on probability."""
    if probability < 0.3:
        return "LOW"
    elif probability < 0.6:
        return "MODERATE"
    else:
        return "HIGH"


def create_error_response(error_message):
    """Create a standardized error response."""
    return {
        "prediction": 0,
        "probability": 0.0,
        "risk_level": "ERROR",
        "interpretation": f"Error: {error_message}",
        "model_used": None,
    }


def process_prediction_result(prediction_result):
    """Process a prediction result into a standardized format."""
    # Check for errors
    if "error" in prediction_result:
        return create_error_response(prediction_result["error"])

    # Get model information
    model_used = prediction_result.get("model_used")
    if not model_used:
        logger.error("No model_used field in prediction result")
        return create_error_response("No model available for prediction")

    # Get prediction and probability based on model used
    prediction = None
    probability = None

    if model_used == "ensemble" and "ensemble_predictions" in prediction_result:
        prediction = prediction_result["ensemble_predictions"][0]
        probability = prediction_result.get("ensemble_probabilities", [0.5])[0]
    elif model_used == "sklearn_mlp" and "sklearn_predictions" in prediction_result:
        prediction = prediction_result["sklearn_predictions"][0]
        probability = prediction_result.get("sklearn_probabilities", [0.5])[0]
    elif model_used == "keras_mlp" and "keras_predictions" in prediction_result:
        prediction = prediction_result["keras_predictions"][0]
        probability = prediction_result.get("keras_probabilities", [0.5])[0]
    else:
        logger.error(f"Missing prediction data for model {model_used}")
        return create_error_response("No prediction data available")

    # Create response
    return {
        "prediction": int(prediction),
        "probability": float(probability),
        "risk_level": get_risk_level(float(probability)),
        "interpretation": prediction_result.get("interpretation", "No interpretation available"),
        "model_used": model_used,
    }


def process_patient_chunk(patients_chunk, model_name=None):
    """Process a chunk of patients in parallel."""
    chunk_results = []

    for patient_dict in patients_chunk:
        try:
            # Make prediction for this patient with caching
            prediction_result = model_predictor.predict(
                patient_dict,
                return_probabilities=True,
                return_interpretation=True,
                model=model_name,
                use_cache=True,
            )

            # Process the prediction result
            result = process_prediction_result(prediction_result)
            chunk_results.append(result)

        except Exception as e:
            logger.error(f"Error processing patient: {str(e)}")
            chunk_results.append(create_error_response(str(e)))

    return chunk_results


async def process_batch_optimized(patients_data, model_name=None):
    """Process a batch of patients using chunking and parallel processing."""
    start_time = time.time()

    # Convert patients data to dictionaries
    patient_dicts = [
        patient.model_dump() if hasattr(patient, "model_dump") else patient.dict()
        for patient in patients_data
    ]

    # Split into chunks
    chunks = [
        patient_dicts[i : i + BATCH_SIZE]  # noqa: E203
        for i in range(0, len(patient_dicts), BATCH_SIZE)
    ]

    # Process chunks in parallel
    loop = asyncio.get_event_loop()
    tasks = [
        loop.run_in_executor(thread_pool, process_patient_chunk, chunk, model_name)
        for chunk in chunks
    ]

    # Wait for all chunks to complete
    chunk_results = await asyncio.gather(*tasks)

    # Flatten results
    all_results = [result for chunk in chunk_results for result in chunk]

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

    return all_results, performance_metrics


@router.post("/auth/token")
async def get_access_token(api_key: str = Depends(api_key_scheme)):
    """Get an access token for API authentication."""
    if not auth_settings.enabled:
        return auth_handler.create_access_token()
    if not api_key or not auth_handler.verify_authentication(api_key):
        raise AuthenticationError("Invalid API key")
    return auth_handler.create_access_token()


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: Request,
    patient_data: PatientData,
    model: str = None,
    authenticated: bool = Depends(verify_authentication),
):
    """Predict heart disease risk for a single patient."""
    logger.info("Starting prediction request")
    try:
        # Debug model predictor state
        if not hasattr(request.app.state, "model_predictor"):
            logger.error("Model predictor not found in app state")
            raise AttributeError("Model predictor not found in app state")

        model_predictor = request.app.state.model_predictor
        logger.info(
            f"Model predictor loaded: "
            f"sklearn={model_predictor.has_sklearn_model}, "
            f"keras={model_predictor.has_keras_model}"
        )

        # Convert to dictionary
        patient_dict = (
            patient_data.model_dump()
            if hasattr(patient_data, "model_dump")
            else patient_data.dict()
        )

        # Make prediction
        result = model_predictor.predict(
            patient_dict,
            return_probabilities=True,
            return_interpretation=True,
            model=model,
            use_cache=True,
        )

        # Process the prediction result
        if "error" in result:
            logger.error(f"Prediction error: {result['error']}")
            raise HTTPException(status_code=500, detail=f"Error: {result['error']}")

        # Map the results to the expected format
        processed_result = process_prediction_result(result)
        logger.info(f"Prediction response prepared: {processed_result}")

        return {
            "prediction": processed_result["prediction"],
            "probability": processed_result["probability"],
            "interpretation": processed_result["interpretation"],
            "risk_level": processed_result["risk_level"],
        }

    except AttributeError as e:
        logger.error(f"State error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server configuration error: {str(e)}")
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    patients_data: List[PatientData],
    model: str = None,
    authenticated: bool = Depends(verify_authentication),
):
    """
    Predict heart disease risk for multiple patients in a batch.

    This endpoint processes multiple patient records in parallel for improved performance.
    It returns predictions along with performance metrics.
    """
    logger.info(f"Received batch prediction request for {len(patients_data)} patients")

    try:
        # Validate the batch request
        validate_batch_request(
            [p.model_dump() if hasattr(p, "model_dump") else p.dict() for p in patients_data]
        )

        # Process batch with optimized method
        predictions, performance_metrics = await process_batch_optimized(patients_data, model)

        # Log performance metrics if enabled
        if PERFORMANCE_LOGGING:
            logger.info(
                f"Batch prediction completed: {len(predictions)} patients processed in "
                f"{performance_metrics['processing_time_seconds']:.2f}s "
                f"({performance_metrics['throughput_patients_per_second']:.2f} patients/sec)"
            )

        return {"predictions": predictions, "performance_metrics": performance_metrics}

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error making batch predictions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing batch prediction: {str(e)}",
        )


@router.get("/models/info")
async def get_model_info(authenticated: bool = Depends(verify_authentication)):
    """Get information about available models."""
    try:
        models_available = {
            "sklearn_mlp": model_predictor.has_sklearn_model,
            "keras_mlp": model_predictor.has_keras_model,
        }

        return {
            "models_available": models_available,
            "ensemble_available": model_predictor.has_ensemble_model,
            "preprocessor_available": model_predictor.preprocessor is not None,
        }

    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")


@router.get("/version")
async def get_version():
    """Get API and model versions."""
    data = {
        "api_version": os.getenv("API_VERSION", "1.0.0"),
        "model_version": model_predictor.model_version
        if hasattr(model_predictor, "model_version")
        else "unknown",
        "environment": os.getenv("ENVIRONMENT", "development"),
    }
    return data


@router.get("/metrics")
async def get_metrics():
    """Get API metrics."""
    data = {
        "uptime_seconds": int(time.time() - SERVER_START_TIME),
        "server_start": datetime.fromtimestamp(SERVER_START_TIME).isoformat(),
        "status": "ok",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "total_predictions": getattr(model_predictor, "total_predictions", 0),
    }
    return data


class CacheConfig(BaseModel):
    enabled: bool = Field(..., description="Whether the cache is enabled")
    max_size: int = Field(..., description="Maximum number of items in cache")
    ttl: int = Field(..., description="Time to live in seconds")


@router.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics."""
    try:
        return model_predictor.get_cache_stats()
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get cache stats: {str(e)}",
        )


@router.post("/cache/config")
async def configure_cache(config: CacheConfig):
    """Configure cache settings."""
    try:
        if config.max_size < 1:
            raise ValueError("Cache max_size must be greater than 0")
        if config.ttl < 0:
            raise ValueError("Cache TTL must be non-negative")

        model_predictor.configure_cache(
            enabled=config.enabled, max_size=config.max_size, ttl=config.ttl
        )
        return {
            "status": "ok",
            "message": "Cache configured successfully",
            "config": {"enabled": config.enabled, "max_size": config.max_size, "ttl": config.ttl},
        }
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        logger.error(f"Error configuring cache: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to configure cache: {str(e)}",
        )


@router.post("/cache/clear")
async def clear_cache():
    """Clear the prediction cache."""
    try:
        model_predictor.clear_cache()
        return {"status": "ok", "message": "Cache cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear cache: {str(e)}",
        )
