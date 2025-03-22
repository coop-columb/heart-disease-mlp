"""
Module for making predictions with trained heart disease models.
"""
# flake8: noqa: E501
import logging
import os

import joblib
import pandas as pd
from tensorflow import keras

from src.models.mlp_model import combine_predictions, interpret_prediction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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
                logger.info("Loaded scikit-learn model")
            else:
                logger.warning(f"scikit-learn model not found at {sklearn_path}")

            # Load Keras model
            keras_path = os.path.join(self.model_dir, "keras_mlp_model.h5")
            if os.path.exists(keras_path):
                self.keras_model = keras.models.load_model(keras_path)
                logger.info("Loaded Keras model")
            else:
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

        # Apply preprocessor if available
        if self.preprocessor is not None:
            processed_data = self.preprocessor.transform(patient_df)
            return processed_data
        else:
            logger.warning("Preprocessor not available. Using raw input data.")
            return patient_df.values

    def predict(
        self, patient_data, return_probabilities=True, return_interpretation=False
    ):
        """
        Make predictions for patient data.

        Args:
            patient_data: Dictionary or DataFrame of patient features
            return_probabilities: Whether to return probability estimates
            return_interpretation: Whether to return clinical interpretation

        Returns:
            Dictionary containing predictions and optionally probabilities
            and interpretation
        """
        logger.info("Making predictions")

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

        # Make predictions with scikit-learn model
        if self.sklearn_model is not None:
            sklearn_probas = self.sklearn_model.predict_proba(X)[:, 1]
            sklearn_preds = (sklearn_probas >= 0.5).astype(int)

            results["sklearn_predictions"] = sklearn_preds
            if return_probabilities:
                results["sklearn_probabilities"] = sklearn_probas

        # Make predictions with Keras model
        if self.keras_model is not None:
            keras_probas = self.keras_model.predict(X).ravel()
            keras_preds = (keras_probas >= 0.5).astype(int)

            results["keras_predictions"] = keras_preds
            if return_probabilities:
                results["keras_probabilities"] = keras_probas

        # Combine predictions if both models are available
        if self.sklearn_model is not None and self.keras_model is not None:
            combined_probas = combine_predictions(
                sklearn_probas, keras_probas, method="mean"
            )
            combined_preds = (combined_probas >= 0.5).astype(int)

            results["ensemble_predictions"] = combined_preds
            if return_probabilities:
                results["ensemble_probabilities"] = combined_probas

            # Use ensemble probabilities for interpretation
            if return_interpretation and original_data is not None:
                interpretation = interpret_prediction(original_data, combined_probas[0])
                results["interpretation"] = interpretation

        # Use available model if only one is loaded
        elif return_interpretation and original_data is not None:
            if self.sklearn_model is not None:
                interpretation = interpret_prediction(original_data, sklearn_probas[0])
            elif self.keras_model is not None:
                interpretation = interpret_prediction(original_data, keras_probas[0])
            else:
                interpretation = "No models available for interpretation."

            results["interpretation"] = interpretation

        return results
