"""
Tests for MLP model functionality.
"""
import numpy as np
import pytest
import tensorflow as tf
from sklearn.neural_network import MLPClassifier

from src.models.mlp_model import (
    build_keras_mlp,
    build_sklearn_mlp,
    combine_predictions,
    evaluate_sklearn_mlp,
    interpret_prediction,
    train_sklearn_mlp,
)


@pytest.fixture
def sample_data():
    """Create a small sample dataset for testing."""
    # Features
    X = np.random.rand(100, 10)
    # Binary target
    y = np.random.randint(0, 2, 100)

    # Split data
    X_train = X[:70]
    X_val = X[70:85]
    X_test = X[85:]
    y_train = y[:70]
    y_val = y[70:85]
    y_test = y[85:]

    return X_train, X_val, X_test, y_train, y_val, y_test


def test_build_sklearn_mlp():
    """Test that scikit-learn MLP model is built correctly."""
    model = build_sklearn_mlp(
        hidden_layer_sizes=(100, 50),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        learning_rate_init=0.001,
        max_iter=1000,
        random_state=42,
    )

    # Check model type
    assert isinstance(model, MLPClassifier)

    # Check model parameters
    assert model.hidden_layer_sizes == (100, 50)
    assert model.activation == "relu"
    assert model.solver == "adam"
    assert model.alpha == 0.0001
    assert model.learning_rate_init == 0.001
    assert model.max_iter == 1000
    assert model.random_state == 42


def test_train_sklearn_mlp(sample_data):
    """Test that scikit-learn MLP model can be trained."""
    X_train, X_val, _, y_train, y_val, _ = sample_data

    # Build model
    model = build_sklearn_mlp(
        hidden_layer_sizes=(5,),  # Small network for quick testing
        max_iter=10,
        random_state=42,
    )

    # Train model without validation data
    trained_model = train_sklearn_mlp(model, X_train, y_train)

    # Check that model was trained
    assert hasattr(trained_model, "classes_")

    # Train model with validation data
    trained_model = train_sklearn_mlp(model, X_train, y_train, X_val, y_val)

    # Check that model was trained
    assert hasattr(trained_model, "classes_")


def test_evaluate_sklearn_mlp(sample_data):
    """Test that scikit-learn MLP model can be evaluated."""
    X_train, _, X_test, y_train, _, y_test = sample_data

    # Build and train model
    model = build_sklearn_mlp(
        hidden_layer_sizes=(5,),  # Small network for quick testing
        max_iter=10,
        random_state=42,
    )
    model = train_sklearn_mlp(model, X_train, y_train)

    # Evaluate model
    metrics, y_pred, y_pred_proba = evaluate_sklearn_mlp(model, X_test, y_test)

    # Check metrics
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics

    # Check predictions
    assert len(y_pred) == len(y_test)
    assert len(y_pred_proba) == len(y_test)


def test_build_keras_mlp():
    """Test that Keras MLP model is built correctly."""
    input_dim = 10
    architecture = [
        {"units": 64, "activation": "relu", "dropout": 0.2, "l2_regularization": 0.01},
        {"units": 32, "activation": "relu", "dropout": 0.1, "l2_regularization": 0.01},
    ]
    
    model = build_keras_mlp(
        input_dim=input_dim,
        architecture=architecture,
        learning_rate=0.001
    )
    
    # Check model type
    assert isinstance(model, tf.keras.Model)
    
    # Check that model has been compiled
    assert model.optimizer is not None
    assert model.loss is not None
    
    # Check that model has the correct architecture layers
    assert len(model.layers) >= len(architecture) + 2  # Input, hidden layers, output


def test_combine_predictions():
    """Test that predictions can be combined correctly."""
    sklearn_proba = np.array([0.1, 0.7, 0.3, 0.9, 0.4])
    keras_proba = np.array([0.2, 0.8, 0.4, 0.8, 0.3])
    
    # Test mean method
    mean_proba = combine_predictions(sklearn_proba, keras_proba, method="mean")
    assert np.allclose(mean_proba, (sklearn_proba + keras_proba) / 2)
    
    # Test max method
    max_proba = combine_predictions(sklearn_proba, keras_proba, method="max")
    assert np.allclose(max_proba, np.maximum(sklearn_proba, keras_proba))
    
    # Test product method
    product_proba = combine_predictions(sklearn_proba, keras_proba, method="product")
    assert np.allclose(product_proba, np.sqrt(sklearn_proba * keras_proba))


def test_interpret_prediction():
    """Test that predictions can be interpreted correctly."""
    # Create a sample patient with risk factors
    patient = {
        "age": 60,
        "sex": 1,
        "trestbps": 160,
        "chol": 250,
        "fbs": 1,
        "thalach": 130,
        "exang": 1
    }
    
    # Test direct prediction interpretation without a model
    result = interpret_prediction(None, patient, probability=0.85)
    
    # Check that interpretation is a string
    assert isinstance(result, str)
    
    # Check for presence of key phrases for high risk
    assert "HIGH RISK" in result
    assert "Advanced age" in result
    assert "Male over 45" in result
    assert "Elevated resting blood pressure" in result
    assert "High cholesterol" in result
    assert "Fasting blood sugar" in result
    assert "Reduced maximum heart rate" in result
    assert "Exercise-induced angina" in result
    
    # Test with low risk probability
    low_risk_patient = {
        "age": 35,
        "sex": 0,
        "trestbps": 120,
        "chol": 190,
        "fbs": 0,
        "thalach": 180,
        "exang": 0
    }
    
    low_risk_result = interpret_prediction(None, low_risk_patient, probability=0.15)
    
    # Check for presence of key phrases for low risk
    assert "LOW RISK" in low_risk_result
    assert "No major risk factors identified" in low_risk_result