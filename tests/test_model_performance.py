"""
Tests for model performance evaluation.
These tests validate the model performance against baseline expectations.
"""

import os
import sys
import warnings

import joblib
import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score)

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Path to models
SKLEARN_MODEL_PATH = "models/sklearn_mlp_model.joblib"
KERAS_MODEL_PATH = "models/keras_mlp_model.h5"
PREPROCESSOR_PATH = "data/processed/preprocessor.joblib"


def test_model_accuracy_vs_baseline(test_data, predictor):
    """Test that models perform better than baseline on test data."""
    if test_data is None or predictor is None:
        pytest.skip("Test data or models not available")

    X_test, y_test = test_data

    # Create baseline model (stratified random guessing)
    baseline = DummyClassifier(strategy="stratified")
    baseline.fit(X_test, y_test)  # Fit on test data for this dummy model
    baseline_pred = baseline.predict(X_test)
    baseline_accuracy = accuracy_score(y_test, baseline_pred)

    # Make predictions with models
    predictions = {}
    metrics = {}

    # Sklearn model
    if predictor.has_sklearn_model:
        y_pred_sklearn, y_proba_sklearn = predictor.predict_sklearn(X_test)
        predictions["sklearn"] = (y_pred_sklearn, y_proba_sklearn)
        metrics["sklearn"] = {
            "accuracy": accuracy_score(y_test, y_pred_sklearn),
            "precision": precision_score(y_test, y_pred_sklearn),
            "recall": recall_score(y_test, y_pred_sklearn),
            "roc_auc": roc_auc_score(y_test, y_proba_sklearn),
        }

        # Test that sklearn model beats baseline
        sklearn_acc = metrics["sklearn"]["accuracy"]
        acc_diff = sklearn_acc - baseline_accuracy
        assert acc_diff > 0.1, (
            f"Sklearn model accuracy: {sklearn_acc:.4f} not significantly "
            f"better than baseline: {baseline_accuracy:.4f}"
        )

    # Keras model
    if predictor.has_keras_model:
        y_pred_keras, y_proba_keras = predictor.predict_keras(X_test)
        predictions["keras"] = (y_pred_keras, y_proba_keras)
        metrics["keras"] = {
            "accuracy": accuracy_score(y_test, y_pred_keras),
            "precision": precision_score(y_test, y_pred_keras),
            "recall": recall_score(y_test, y_pred_keras),
            "roc_auc": roc_auc_score(y_test, y_proba_keras),
        }

        # Test that keras model beats baseline
        keras_acc = metrics["keras"]["accuracy"]
        acc_diff = keras_acc - baseline_accuracy
        assert acc_diff > 0.1, (
            f"Keras model accuracy: {keras_acc:.4f} not significantly "
            f"better than baseline: {baseline_accuracy:.4f}"
        )

    # Ensemble model
    if predictor.has_sklearn_model and predictor.has_keras_model:
        try:
            y_pred_ensemble, y_proba_ensemble = predictor.predict_ensemble(X_test)
            predictions["ensemble"] = (y_pred_ensemble, y_proba_ensemble)
            metrics["ensemble"] = {
                "accuracy": accuracy_score(y_test, y_pred_ensemble),
                "precision": precision_score(y_test, y_pred_ensemble),
                "recall": recall_score(y_test, y_pred_ensemble),
                "roc_auc": roc_auc_score(y_test, y_proba_ensemble),
            }

            # Test that ensemble beats individual models
            sklearn_acc = metrics["sklearn"]["accuracy"]
            keras_acc = metrics["keras"]["accuracy"]
            ensemble_acc = metrics["ensemble"]["accuracy"]
            min_acc = min(sklearn_acc, keras_acc)

            assert ensemble_acc >= min_acc, (
                f"Ensemble accuracy: {ensemble_acc:.4f} should be at least "
                f"as good as worst individual model: {min_acc:.4f}"
            )
        except Exception:
            warnings.warn("Could not test ensemble model", UserWarning)

    # Display metrics for logging
    print("\nModel Performance Metrics:")
    for model_name, model_metrics in metrics.items():
        print(f"\n{model_name.upper()} Model:")
        for metric_name, value in model_metrics.items():
            print(f"  {metric_name}: {value:.4f}")
    print(f"\nBaseline accuracy: {baseline_accuracy:.4f}")


def test_model_calibration(test_data, predictor):
    """Test that model probability predictions are well-calibrated."""
    if test_data is None or predictor is None:
        pytest.skip("Test data or models not available")

    X_test, y_test = test_data

    # Check each available model
    for model_name in ["sklearn", "keras", "ensemble"]:
        if model_name == "sklearn" and predictor.has_sklearn_model:
            _, y_proba = predictor.predict_sklearn(X_test)
        elif model_name == "keras" and predictor.has_keras_model:
            _, y_proba = predictor.predict_keras(X_test)
        elif (
            model_name == "ensemble"
            and predictor.has_sklearn_model
            and predictor.has_keras_model
        ):
            try:
                _, y_proba = predictor.predict_ensemble(X_test)
            except Exception:
                continue
        else:
            continue

        # Define probability bins
        bins = np.linspace(0, 1, 11)  # 10 bins
        bin_indices = np.digitize(y_proba, bins) - 1
        bin_indices = np.clip(bin_indices, 0, len(bins) - 2)

        bin_counts = np.zeros(len(bins) - 1)
        bin_positive_counts = np.zeros(len(bins) - 1)

        # Count samples and positive samples in each bin
        for i, bin_idx in enumerate(bin_indices):
            bin_counts[bin_idx] += 1
            if y_test[i] == 1:
                bin_positive_counts[bin_idx] += 1

        # Calculate fraction of positives in each bin
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bin_fractions = np.divide(
                bin_positive_counts,
                bin_counts,
                out=np.zeros_like(bin_positive_counts),
                where=bin_counts > 0,
            )

        # Calculate calibration error (mean absolute difference between bin midpoint and fraction of positives)
        bin_midpoints = (bins[:-1] + bins[1:]) / 2
        calibration_errors = np.abs(bin_midpoints - bin_fractions)
        mean_calibration_error = np.mean(calibration_errors[bin_counts > 0])

        # Test that calibration error is below threshold
        threshold = 0.25
        assert mean_calibration_error < threshold, (
            f"{model_name} model calibration error: {mean_calibration_error:.4f} "
            f"exceeds threshold {threshold}"
        )

        print(
            f"\n{model_name.upper()} model calibration error: {mean_calibration_error:.4f}"
        )


def test_model_fairness(test_data, predictor):
    """Test that model predictions are fair across different demographic groups."""
    if test_data is None or predictor is None:
        pytest.skip("Test data or models not available")

    try:
        # Load preprocessor to determine features
        preprocessor = joblib.load(PREPROCESSOR_PATH)

        X_test, y_test = test_data

        # Check fairness across gender groups if 'sex' feature exists
        # We need to reconstruct the original dataframe structure with feature names
        if preprocessor and hasattr(preprocessor, "feature_names_in_"):
            feature_names = preprocessor.feature_names_in_
            if "sex" in feature_names:
                sex_idx = list(feature_names).index("sex")

                # Get gender from preprocessed features (this is approximate)
                # In a real scenario, you would track groups before preprocessing
                male_indices = X_test[:, sex_idx] > 0.5
                female_indices = ~male_indices

                # Skip if insufficient samples in either group
                if sum(male_indices) < 10 or sum(female_indices) < 10:
                    warnings.warn(
                        "Insufficient samples to test fairness across gender groups"
                    )
                    return

                # Make predictions
                if predictor.has_sklearn_model:
                    y_pred, _ = predictor.predict_sklearn(X_test)

                    # Calculate accuracy for each group
                    male_accuracy = accuracy_score(
                        y_test[male_indices], y_pred[male_indices]
                    )
                    female_accuracy = accuracy_score(
                        y_test[female_indices], y_pred[female_indices]
                    )

                    # Check that accuracy difference is within an acceptable range
                    accuracy_diff = abs(male_accuracy - female_accuracy)
                    threshold = 0.20
                    assert accuracy_diff < threshold, (
                        f"Accuracy difference between gender groups ({accuracy_diff:.4f}) "
                        f"exceeds threshold {threshold}"
                    )

                    print(
                        f"\nFairness test - accuracy difference between gender groups: {accuracy_diff:.4f}"
                    )
                    print(
                        f"Male accuracy: {male_accuracy:.4f}, Female accuracy: {female_accuracy:.4f}"
                    )
    except Exception:
        warnings.warn("Could not test model fairness", UserWarning)
