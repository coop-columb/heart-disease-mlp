"""
Module for making predictions with trained heart disease models.
"""

# flake8: noqa: E501
import datetime
import hashlib
import json
import logging
import os
import time
from collections import OrderedDict

import joblib
import numpy as np
import pandas as pd
from cachetools import TTLCache
from tensorflow import keras

from models.mlp_model import combine_predictions, interpret_prediction
from utils import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load configuration
config = load_config()
cache_config = config.get("api", {}).get("caching", {})
CACHE_ENABLED = cache_config.get("enabled", True)
CACHE_MAX_SIZE = cache_config.get("max_size", 1000)
CACHE_TTL = cache_config.get("ttl", 3600)  # 1 hour default
CACHE_HASH_ALGORITHM = cache_config.get("hash_algorithm", "md5")


class PredictionCache:
    """Cache for model predictions."""

    def __init__(self, max_size=CACHE_MAX_SIZE, ttl=CACHE_TTL):
        """Initialize the cache with size and TTL settings."""
        self.max_size = max_size
        self.ttl = ttl
        self.cache = TTLCache(maxsize=max_size, ttl=ttl)
        self.hits = 0
        self.misses = 0
        self.enabled = True

    def _make_key(self, data, model=None):
        """Create a cache key from input data and model type."""
        try:
            if isinstance(data, dict):
                # Sort dictionary items to ensure consistent key generation
                data_str = json.dumps(data, sort_keys=True)
            else:
                # Convert DataFrame or array to string
                data_str = str(data)
            return f"{data_str}:{model or 'default'}"
        except Exception as e:
            logger.error(f"Error creating cache key: {e}")
            return None

    def get(self, data, model=None):
        """Get prediction from cache."""
        if not self.enabled:
            return None

        key = self._make_key(data, model)
        if not key:
            return None

        result = self.cache.get(key)
        if result is not None:
            self.hits += 1
        else:
            self.misses += 1
        return result

    def put(self, data, model, result):
        """Store prediction in cache."""
        if not self.enabled:
            return

        key = self._make_key(data, model)
        if key:
            self.cache[key] = result

    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        return {"status": "ok", "message": "Cache cleared successfully"}

    def get_stats(self):
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        return {
            "enabled": self.enabled,
            "max_size": self.max_size,
            "ttl": self.ttl,
            "current_size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
        }


class HeartDiseasePredictor:
    """Heart disease prediction model wrapper."""

    def __init__(self, model_dir="models"):
        """Initialize the predictor with models and cache settings."""
        self.model_dir = model_dir
        self.sklearn_model = None
        self.keras_model = None
        self.preprocessor = None
        self.has_sklearn_model = False
        self.has_keras_model = False
        self.has_ensemble_model = False

        # Initialize cache settings
        self.cache_enabled = True
        self.cache_max_size = 1000  # default max size
        self.cache_ttl = 3600  # default TTL in seconds

        # Initialize cache
        self.cache = PredictionCache(max_size=CACHE_MAX_SIZE, ttl=CACHE_TTL)

        self.total_predictions = 0
        self.cache_hits = 0
        self.cache_misses = 0

        # Load models
        self.load_models()

    def load_models(self):
        """Load trained models from disk."""
        logger.info(f"Loading models from {self.model_dir}")

        # Log the absolute paths
        sklearn_path = os.path.join(self.model_dir, "sklearn_mlp_model.joblib")
        keras_path = os.path.join(self.model_dir, "keras_mlp_model.h5")
        preprocessor_path = os.path.join("data/processed/preprocessor.joblib")

        logger.info(f"Attempting to load scikit-learn model from: {os.path.abspath(sklearn_path)}")
        logger.info(f"Attempting to load Keras model from: {os.path.abspath(keras_path)}")
        logger.info(f"Attempting to load preprocessor from: {os.path.abspath(preprocessor_path)}")

        try:
            # Load scikit-learn model
            if os.path.exists(sklearn_path):
                logger.info("Found scikit-learn model file")
                self.sklearn_model = joblib.load(sklearn_path)
                self.has_sklearn_model = True
                logger.info("Successfully loaded scikit-learn model")
            else:
                self.has_sklearn_model = False
                logger.warning(f"scikit-learn model not found at {sklearn_path}")

            # Load Keras model
            if os.path.exists(keras_path):
                logger.info("Found Keras model file")
                self.keras_model = keras.models.load_model(keras_path)
                self.has_keras_model = True
                logger.info("Successfully loaded Keras model")
            else:
                self.has_keras_model = False
                logger.warning(f"Keras model not found at {keras_path}")

            # Load preprocessor
            if os.path.exists(preprocessor_path):
                logger.info("Found preprocessor file")
                self.preprocessor = joblib.load(preprocessor_path)
                logger.info("Successfully loaded preprocessor")
            else:
                logger.warning(f"Preprocessor not found at {preprocessor_path}")
                self.preprocessor = None

        except Exception as e:
            logger.error(f"Error loading models: {str(e)}", exc_info=True)
            raise

    def preprocess_input(self, patient_data):
        """
        Preprocess patient data for prediction.

        Args:
            patient_data: Dictionary or DataFrame of patient features

        Returns:
            Preprocessed feature array
        """
        logger.info("Preprocessing input data")

        # Convert dictionary to DataFrame if needed
        if isinstance(patient_data, dict):
            patient_df = pd.DataFrame([patient_data])
        else:
            patient_df = patient_data

        # Convert to numpy array to remove feature names before preprocessing
        input_data = patient_df.to_numpy()

        # Apply preprocessor if available
        if self.preprocessor is not None:
            processed_data = self.preprocessor.transform(input_data)
            return processed_data
        else:
            logger.warning("Preprocessor not available. Using raw input data.")
            return input_data

    def predict(
        self,
        patient_data,
        return_probabilities=True,
        return_interpretation=False,
        model=None,
        use_cache=True,
    ):
        """
        Make predictions for patient data.

        Args:
            patient_data: Dictionary or DataFrame of patient features
            return_probabilities: Whether to return probability estimates
            return_interpretation: Whether to return clinical interpretation
            model: Optional specific model to use (sklearn, keras, ensemble)
            use_cache: Whether to use the prediction cache

        Returns:
            Dictionary containing predictions and optionally probabilities
            and interpretation
        """
        # Check cache first if enabled
        # Check cache first if enabled
        if use_cache and self.cache.enabled:
            # Try to get from cache
            cached_result = self.cache.get(patient_data, model)
            if cached_result is not None:
                logger.info("Using cached prediction result")

                # Make sure we include interpretation if requested and available
                if (
                    return_interpretation
                    and "interpretation" not in cached_result
                    and isinstance(patient_data, dict)
                ):
                    # Add interpretation if not in cached result
                    model_used = cached_result.get("model_used", "ensemble")
                    probability = None

                    if model_used == "ensemble" and "ensemble_probabilities" in cached_result:
                        probability = cached_result["ensemble_probabilities"][0]
                    elif model_used == "sklearn_mlp" and "sklearn_probabilities" in cached_result:
                        probability = cached_result["sklearn_probabilities"][0]
                    elif model_used == "keras_mlp" and "keras_probabilities" in cached_result:
                        probability = cached_result["keras_probabilities"][0]

                    if probability is not None:
                        cached_result["interpretation"] = interpret_prediction(
                            patient_data=patient_data, probability=probability
                        )

                return cached_result

        logger.info("Making predictions")

        try:
            # Store original data for interpretation
            if isinstance(patient_data, dict):
                original_data = patient_data
            else:
                original_data = patient_data.iloc[0].to_dict() if len(patient_data) == 1 else None

            # Preprocess input
            X = self.preprocess_input(patient_data)

            # Initialize results
            results = {}

            # Track which model predictions are available
            sklearn_available = False
            keras_available = False
            sklearn_probas = None
            keras_probas = None

            # Track which model was actually used for caching
            model_used = None

            # Make predictions based on requested model or available models
            # If specific model requested, only try that one
            if model == "sklearn" and self.sklearn_model is not None:
                try:
                    sklearn_probas = self.sklearn_model.predict_proba(X)[:, 1]
                    sklearn_preds = (sklearn_probas >= 0.5).astype(int)
                    results["sklearn_predictions"] = sklearn_preds
                    if return_probabilities:
                        results["sklearn_probabilities"] = sklearn_probas
                    sklearn_available = True
                    model_used = "sklearn_mlp"
                except Exception as e:
                    logger.warning(f"Error making scikit-learn predictions: {e}")
            elif model == "keras" and self.keras_model is not None:
                try:
                    # Get predictions with proper tensor conversion
                    keras_pred_raw = self.keras_model.predict(X, verbose=0)
                    keras_probas = np.asarray(keras_pred_raw).reshape(-1)
                    # Ensure we use numpy array operations
                    if keras_probas.size == 1:
                        keras_probas = np.array([keras_probas[0].item()])
                    keras_preds = (keras_probas >= 0.5).astype(int)
                    results["keras_predictions"] = keras_preds
                    if return_probabilities:
                        results["keras_probabilities"] = keras_probas
                    keras_available = True
                    model_used = "keras_mlp"
                except Exception as e:
                    logger.warning(f"Error making Keras predictions: {e}")
            # If ensemble is requested or no specific model, try both
            else:
                # Try scikit-learn
                if self.sklearn_model is not None:
                    try:
                        sklearn_probas = self.sklearn_model.predict_proba(X)[:, 1]
                        sklearn_preds = (sklearn_probas >= 0.5).astype(int)
                        results["sklearn_predictions"] = sklearn_preds
                        if return_probabilities:
                            results["sklearn_probabilities"] = sklearn_probas
                        sklearn_available = True
                    except Exception as e:
                        logger.warning(f"Error making scikit-learn predictions: {e}")

                # Try Keras
                if self.keras_model is not None:
                    try:
                        # Get predictions with proper tensor conversion
                        keras_pred_raw = self.keras_model.predict(X, verbose=0)
                        keras_probas = np.asarray(keras_pred_raw).reshape(-1)
                        # Ensure we use numpy array operations
                        if keras_probas.size == 1:
                            keras_probas = np.array([keras_probas[0].item()])
                        keras_preds = (keras_probas >= 0.5).astype(int)
                        results["keras_predictions"] = keras_preds
                        if return_probabilities:
                            results["keras_probabilities"] = keras_probas
                        keras_available = True
                        model_used = "keras_mlp"
                    except Exception as e:
                        logger.warning(f"Error making Keras predictions: {e}")

            # Use ensemble if both are available and no specific model was requested
            # or ensemble was specifically requested
            if (model is None or model == "ensemble") and sklearn_available and keras_available:
                try:
                    combined_probas = combine_predictions(
                        sklearn_probas, keras_probas, method="mean"
                    )
                    combined_preds = (combined_probas >= 0.5).astype(int)

                    self.has_ensemble_model = True
                    results["ensemble_predictions"] = combined_preds
                    if return_probabilities:
                        results["ensemble_probabilities"] = combined_probas
                    model_used = "ensemble"

                    # Use ensemble probabilities for interpretation
                    if return_interpretation and original_data is not None:
                        interpretation = interpret_prediction(
                            patient_data=original_data, probability=combined_probas[0]
                        )
                        results["interpretation"] = interpretation
                except Exception as e:
                    logger.warning(f"Error combining predictions: {e}")
                    self.has_ensemble_model = False
                    # Fall back to available model
                    if sklearn_available:
                        model_used = "sklearn_mlp"
                    elif keras_available:
                        model_used = "keras_mlp"

            # Use available model if ensemble not used and interpretation is requested
            elif return_interpretation and original_data is not None:
                try:
                    if sklearn_available:
                        interpretation = interpret_prediction(
                            patient_data=original_data, probability=sklearn_probas[0]
                        )
                        if not model_used:
                            model_used = "sklearn_mlp"
                    elif keras_available:
                        interpretation = interpret_prediction(
                            patient_data=original_data, probability=keras_probas[0]
                        )
                        if not model_used:
                            model_used = "keras_mlp"
                    else:
                        interpretation = "No models available for interpretation."

                    results["interpretation"] = interpretation
                except Exception as e:
                    logger.warning(f"Error generating interpretation: {e}")
                    results["interpretation"] = "Error generating interpretation."

            # No model is available
            if not sklearn_available and not keras_available:
                results["error"] = "No models available for prediction."
                if return_interpretation:
                    results["interpretation"] = "No models available for interpretation."
            else:
                # Add model_used to results
                results["model_used"] = model_used

                # Cache the result if we have a valid prediction and caching is enabled
                # Cache the result if we have a valid prediction and caching is enabled
                if use_cache and self.cache.enabled and model_used is not None:
                    self.cache.put(patient_data, model, results)
            return results

        except Exception as e:
            logger.error(f"Error in prediction pipeline: {e}")
            # Return graceful error
            return {
                "error": f"Prediction failed: {str(e)}",
                "interpretation": (
                    "Unable to make prediction due to an error." if return_interpretation else None
                ),
            }

    def get_cache_stats(self):
        """
        Get cache statistics.

        Returns:
            Dictionary of cache statistics
        """
        return self.cache.get_stats()

    def clear_cache(self):
        """
        Clear the prediction cache.

        Returns:
            Status message
        """
        self.cache.clear()
        return {"status": "success", "message": "Cache cleared successfully"}

    def configure_cache(self, enabled: bool, max_size: int, ttl: int) -> None:
        """Configure the prediction cache.

        Args:
            enabled (bool): Whether to enable the cache
            max_size (int): Maximum number of items in cache
            ttl (int): Time to live in seconds
        """
        global CACHE_ENABLED
        CACHE_ENABLED = enabled
        self.cache.enabled = enabled
        self.cache.max_size = max_size
        self.cache.ttl = ttl

        # Clear existing cache
        self.cache.clear()

        # Update cache parameters
        self.cache.max_size = max_size
        self.cache.ttl = ttl

        # Create new cache with updated settings
        from functools import lru_cache

        @lru_cache(maxsize=max_size)
        def cached_predict(data_key):
            return self._make_prediction(data_key)

        # Create _make_prediction method if it doesn't exist
        if not hasattr(self, "_make_prediction"):

            def _make_prediction(data_key):
                # Convert the key back to original data format and make prediction
                # This is a simplification, actual implementation depends on how
                # data_key represents the input data
                data = json.loads(data_key) if isinstance(data_key, str) else data_key
                return self.predict(data, use_cache=False)

            self._make_prediction = _make_prediction

        self.cached_predict = cached_predict if enabled else self._make_prediction
