import asyncio
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer

from src.heart_api.core import (BATCH_SIZE, MAX_WORKERS, PERFORMANCE_LOGGING,
                                auth_handler, auth_settings, model_predictor,
                                thread_pool)
from src.heart_api.models import (BatchPredictionResponse, PatientData,
                                  PredictionResponse)

logger = logging.getLogger(__name__)
router = APIRouter()

# Setup authentication schemes
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token", auto_error=False)
api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


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

    if api_key and await auth_handler.verify_api_key(api_key):
        return True

    if token and await auth_handler.verify_token(token):
        return True

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


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

            # Get prediction results
            prediction = None
            probability = None
            model_used = prediction_result.get("model_used")

            if model_used and model_used in prediction_result:
                prediction = prediction_result[f"{model_used}_predictions"][0]
                probability = prediction_result.get(
                    f"{model_used}_probabilities", [None]
                )[0]

            # Handle missing results
            if prediction is None or model_used is None:
                chunk_results.append(
                    {
                        "prediction": 0,
                        "probability": 0.0,
                        "risk_level": "ERROR",
                        "interpretation": "No predictions available",
                        "model_used": None,
                    }
                )
                continue

            # Use default probability if missing
            if probability is None:
                probability = 0.5

            # Determine risk level
            if probability < 0.3:
                risk_level = "LOW"
            elif probability < 0.6:
                risk_level = "MODERATE"
            else:
                risk_level = "HIGH"

            # Get interpretation
            interpretation = prediction_result.get("interpretation")

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
            logger.error(f"Error processing patient: {str(e)}")
            chunk_results.append(
                {
                    "prediction": 0,
                    "probability": 0.0,
                    "risk_level": "ERROR",
                    "interpretation": f"Error: {str(e)}",
                    "model_used": None,
                }
            )

    return chunk_results


async def process_batch_optimized(patients_data, model_name=None):
    """Process a batch of patients using chunking and parallel processing."""
    import time

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
async def get_access_token():
    """Get an access token for API authentication."""
    return auth_handler.create_access_token()


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    patient_data: PatientData,
    model: str = None,
    authenticated: bool = Depends(verify_authentication),
):
    """Predict heart disease risk for a single patient."""
    try:
        # Convert to dictionary
        patient_dict = (
            patient_data.model_dump()
            if hasattr(patient_data, "model_dump")
            else patient_data.dict()
        )

        # Make prediction
        prediction_result = model_predictor.predict(
            patient_dict,
            return_probabilities=True,
            return_interpretation=True,
            model=model,
            use_cache=True,
        )

        if "error" in prediction_result:
            raise HTTPException(status_code=500, detail=prediction_result["error"])

        # Process prediction results
        model_used = prediction_result.get("model_used")
        if not model_used:
            raise HTTPException(
                status_code=500, detail="No model available for prediction"
            )

        # Get prediction and probability
        prediction = prediction_result[f"{model_used}_predictions"][0]
        probability = prediction_result.get(f"{model_used}_probabilities", [0.5])[0]

        # Determine risk level
        if probability < 0.3:
            risk_level = "LOW"
        elif probability < 0.6:
            risk_level = "MODERATE"
        else:
            risk_level = "HIGH"

        return {
            "prediction": int(prediction),
            "probability": float(probability),
            "risk_level": risk_level,
            "interpretation": prediction_result.get("interpretation"),
            "model_used": model_used,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    patients_data: List[PatientData],
    model: str = None,
    authenticated: bool = Depends(verify_authentication),
):
    """Predict heart disease risk for multiple patients."""
    try:
        if not patients_data:
            raise ValueError("No patient data provided")

        results, performance_metrics = await process_batch_optimized(
            patients_data, model
        )

        if not results:
            raise ValueError("No valid predictions could be made")

        response = {"predictions": results}
        if PERFORMANCE_LOGGING:
            response["performance_metrics"] = performance_metrics

        return response

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error making batch predictions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


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
        raise HTTPException(
            status_code=500, detail=f"Error getting model info: {str(e)}"
        )
