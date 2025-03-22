"""
Tests for data processing functionality.
"""

import os

import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import (binarize_target,
                                 create_preprocessing_pipeline,
                                 handle_missing_values, load_data, split_data)
from src.features.feature_engineering import (create_feature_interactions,
                                              create_medical_risk_score)


@pytest.fixture
def sample_data():
    """Create a small sample dataset for testing."""
    data = {
        "age": [45, 61, 52, 39, 58],
        "sex": [1, 0, 1, 0, 1],
        "cp": [3, 2, 1, 4, 2],
        "trestbps": [140, 130, 125, 110, 150],
        "chol": [220, 240, 200, 190, 270],
        "fbs": [0, 1, 0, 0, 1],
        "restecg": [1, 2, 0, 1, 2],
        "thalach": [170, 150, 165, 180, 145],
        "exang": [0, 1, 0, 0, 1],
        "oldpeak": [1.2, 2.5, 0.8, 0.5, 3.0],
        "slope": [2, 1, 2, 3, 1],
        "ca": [0, 2, 1, 0, 3],
        "thal": [3, 7, 3, 6, 7],
        "target": [0, 1, 0, 0, 1],
    }
    return pd.DataFrame(data)


def test_binarize_target(sample_data):
    """Test that binarize_target correctly converts target to binary."""
    # Create a sample with non-binary target
    df = sample_data.copy()
    df["target"] = [0, 1, 2, 3, 4]

    # Binarize target
    result = binarize_target(df)

    # Check that target is now binary
    assert set(result["target"].unique()) == {0, 1}
    assert result.loc[0, "target"] == 0  # First row had target=0
    assert all(result.loc[1:, "target"] == 1)  # All others should be 1


def test_handle_missing_values():
    """Test that missing values are handled correctly."""
    # Create data with missing values
    data = {
        "age": [45, 61, np.nan, 39, 58],
        "sex": [1, 0, 1, np.nan, 1],
        "chol": [220, np.nan, 200, 190, 270],
    }
    df = pd.DataFrame(data)

    # Handle missing values
    result = handle_missing_values(df)

    # Check that there are no missing values
    assert result.isnull().sum().sum() == 0

    # Check that correct imputation was used
    assert result.loc[2, "age"] == df["age"].median()
    assert result.loc[3, "sex"] == 1  # Most frequent
    assert result.loc[1, "chol"] == df["chol"].median()


def test_create_preprocessing_pipeline():
    """Test that preprocessing pipeline is created correctly."""
    categorical_features = ["sex", "cp"]
    numerical_features = ["age", "trestbps", "chol"]

    # Create pipeline
    preprocessor = create_preprocessing_pipeline(
        categorical_features, numerical_features, "median", True
    )

    # Check that pipeline has correct transformers
    assert len(preprocessor.transformers) == 2

    # Check transformer names
    transformer_names = [name for name, _, _ in preprocessor.transformers]
    assert "num" in transformer_names
    assert "cat" in transformer_names


def test_split_data(sample_data):
    """Test that data is split correctly."""
    X = sample_data.drop("target", axis=1)
    y = sample_data["target"]

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, test_size=0.2, val_size=0.25, random_state=42
    )

    # Check shapes
    assert len(X_train) + len(X_val) + len(X_test) == len(X)
    assert len(y_train) + len(y_val) + len(y_test) == len(y)

    # Check that test set is 20% of data
    assert len(X_test) == int(0.2 * len(X))

    # Check that validation set is 25% of training data
    assert len(X_val) == int(0.25 * (len(X) - len(X_test)))


def test_create_feature_interactions(sample_data):
    """Test that feature interactions are created correctly."""
    # Create feature interactions
    result = create_feature_interactions(sample_data)

    # Check that new features are created
    assert "age_sex" in result.columns
    assert "cp_exang" in result.columns
    assert "bp_chol" in result.columns

    # Check values
    assert (
        result.loc[0, "age_sex"]
        == sample_data.loc[0, "age"] * sample_data.loc[0, "sex"]
    )
    assert (
        result.loc[0, "cp_exang"]
        == sample_data.loc[0, "cp"] * sample_data.loc[0, "exang"]
    )


def test_create_medical_risk_score(sample_data):
    """Test that medical risk score is created correctly."""
    # Create risk score
    result = create_medical_risk_score(sample_data)

    # Check that risk score column exists
    assert "risk_score" in result.columns

    # Check risk score values
    # The second row (index 1) has high blood pressure, high cholesterol,
    # high blood sugar, and exercise-induced angina
    assert result.loc[1, "risk_score"] >= 3
