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
from tensorflow import keras

from src.models.mlp_model import combine_predictions, interpret_prediction
from src.utils import load_config

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
    """Cache for storing and retrieving prediction results."""

    def __init__(self, max_size=CACHE_MAX_SIZE, ttl=CACHE_TTL):
        """
        Initialize the prediction cache.

        Args:
            max_size: Maximum number of entries in the cache
            ttl: Time-to-live in seconds for cache entries
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache = OrderedDict()  # OrderedDict for LRU behavior
        self.stats = {
            "hits": 0,
            "misses": 0,
            "entries": 0,
            "evictions": 0,
            "created_at": datetime.datetime.now().isoformat(),
        }

    def _generate_key(self, data, model=None):
        """
        Generate a cache key from input data and model.

        Args:
            data: Input data for prediction
            model: Optional model name to use

        Returns:
            String hash key
        """
        # Convert data to a consistent format
        if isinstance(data, dict):
            # Sort to ensure consistent order
            serialized = json.dumps(data, sort_keys=True)
        elif isinstance(data, pd.DataFrame):
            # Convert DataFrame to sorted dict
            serialized = json.dumps(data.to_dict(orient="records"), sort_keys=True)
        else:
            # Convert any other type to string
            serialized = str(data)

        # Add model to the key if provided
        if model:
            serialized += f"_model_{model}"

        # Create hash using configured algorithm
        if CACHE_HASH_ALGORITHM == "md5":
            return hashlib.md5(serialized.encode()).hexdigest()
        elif CACHE_HASH_ALGORITHM == "sha1":
            return hashlib.sha1(serialized.encode()).hexdigest()
        else:
            # Default to md5
            return hashlib.md5(serialized.encode()).hexdigest()

    def get(self, data, model=None):
        """
        Get cached prediction result.

        Args:
            data: Input data for prediction
            model: Optional model name to use

        Returns:
            Cached prediction result or None if not in cache
        """
        if not CACHE_ENABLED:
            return None

        key = self._generate_key(data, model)

        if key in self.cache:
            entry = self.cache[key]
            current_time = time.time()

            # Check if entry is expired
            if current_time - entry["timestamp"] > self.ttl:
                # Remove expired entry
                del self.cache[key]
                self.stats["entries"] -= 1
                self.stats["misses"] += 1
                return None

            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.stats["hits"] += 1
            return entry["result"]

        self.stats["misses"] += 1
        return None

    def put(self, data, model, result):
        """
        Store prediction result in cache.

        Args:
            data: Input data for prediction
            model: Model name used for prediction
            result: Prediction result to cache
        """
        if not CACHE_ENABLED:
            return

        key = self._generate_key(data, model)

        # Create cache entry
        entry = {
            "timestamp": time.time(),
            "result": result,
        }

        # Add to cache
        if key in self.cache:
            # Just update existing entry
            self.cache[key] = entry
            # Move to end (most recently used)
            self.cache.move_to_end(key)
        else:
            # Check if cache is full
            if len(self.cache) >= self.max_size:
                # Remove oldest entry (first item in OrderedDict)
                self.cache.popitem(last=False)
                self.stats["evictions"] += 1
                self.stats["entries"] -= 1

            # Add new entry
            self.cache[key] = entry
            self.stats["entries"] += 1

    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.stats["entries"] = 0
        self.stats["evictions"] = 0
        self.stats["hits"] = 0
        self.stats["misses"] = 0
        self.stats["created_at"] = datetime.datetime.now().isoformat()

    def get_stats(self):
        """
        Get cache statistics.

        Returns:
            Dictionary of cache statistics
        """
        hit_rate = 0
        if (self.stats["hits"] + self.stats["misses"]) > 0:
            hit_rate = self.stats["hits"] / (self.stats["hits"] + self.stats["misses"])

        return {
            "enabled": CACHE_ENABLED,
            "max_size": self.max_size,
            "ttl_seconds": self.ttl,
            "entries": self.stats["entries"],
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_rate": round(hit_rate, 3),
            "evictions": self.stats["evictions"],
            "created_at": self.stats["created_at"],
        }


class HeartDiseasePredictor:
    """Heart disease prediction model wrapper."""

    def __init__(self, model_dir="models"):
        """
        Initialize the predictor with trained models.

        Args:
            model_dir: Directory containing trained models
        """
        self.model_dir = model_dir
        self.sklearn_model = None
        self.keras_model = None
        self.preprocessor = None
        self.has_sklearn_model = False
        self.has_keras_model = False
        self.has_ensemble_model = False

        # Initialize prediction cache
        self.cache = PredictionCache(max_size=CACHE_MAX_SIZE, ttl=CACHE_TTL)

        # Load models
        self.load_models()

    def load_models(self):
        """Load trained models from disk."""
        logger.info(f"Loading models from {self.model_dir}")

        try:
            # Load scikit-learn model
            sklearn_path = os.path.join(self.model_dir, "sklearn_mlp_model.joblib")
            if os.path.exists(sklearn_path):
                self.sklearn_model = joblib.load(sklearn_path)
                self.has_sklearn_model = True
                logger.info("Loaded scikit-learn model")
            else:
                self.has_sklearn_model = False
                logger.warning(f"scikit-learn model not found at {sklearn_path}")

            # Load Keras model
            keras_path = os.path.join(self.model_dir, "keras_mlp_model.h5")
            if os.path.exists(keras_path):
                self.keras_model = keras.models.load_model(keras_path)
                self.has_keras_model = True
                logger.info("Loaded Keras model")
            else:
                self.has_keras_model = False
                logger.warning(f"Keras model not found at {keras_path}")

            # Load preprocessor
            preprocessor_path = os.path.join("data/processed/preprocessor.joblib")
            if os.path.exists(preprocessor_path):
                self.preprocessor = joblib.load(preprocessor_path)
                logger.info("Loaded preprocessor")
            else:
                logger.warning(f"Preprocessor not found at {preprocessor_path}")

        except Exception as e:
            logger.error(f"Error loading models: {e}")
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
        if use_cache and CACHE_ENABLED:
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

                    if (
                        model_used == "ensemble"
                        and "ensemble_probabilities" in cached_result
                    ):
                        probability = cached_result["ensemble_probabilities"][0]
                    elif (
                        model_used == "sklearn_mlp"
                        and "sklearn_probabilities" in cached_result
                    ):
                        probability = cached_result["sklearn_probabilities"][0]
                    elif (
                        model_used == "keras_mlp"
                        and "keras_probabilities" in cached_result
                    ):
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
                original_data = (
                    patient_data.iloc[0].to_dict() if len(patient_data) == 1 else None
                )

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
                        keras_probas = np.array([float(keras_probas[0])])
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
                            keras_probas = np.array([float(keras_probas[0])])
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
            if (
                (model is None or model == "ensemble")
                and sklearn_available
                and keras_available
            ):
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
                    results[
                        "interpretation"
                    ] = "No models available for interpretation."
            else:
                # Add model_used to results
                results["model_used"] = model_used

                # Cache the result if we have a valid prediction and caching is enabled
                if use_cache and CACHE_ENABLED and model_used is not None:
                    self.cache.put(patient_data, model, results)

            return results

        except Exception as e:
            logger.error(f"Error in prediction pipeline: {e}")
            # Return graceful error
            return {
                "error": f"Prediction failed: {str(e)}",
                "interpretation": (
                    "Unable to make prediction due to an error."
                    if return_interpretation
                    else None
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
