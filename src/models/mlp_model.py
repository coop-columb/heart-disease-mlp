"""
MLP model implementations for the Heart Disease Prediction project.
"""

import logging
import os
import sys  # noqa: F401
from typing import Any, Dict, List, Optional, Tuple, Union  # noqa: F401

# Standard libraries for data processing and visualization
import matplotlib.pyplot as plt
import numpy as np

# Machine learning libraries
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from tensorflow import keras
from tensorflow.keras import callbacks, layers, regularizers  # noqa: F401

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def build_sklearn_mlp(
    hidden_layer_sizes=(100, 50),
    activation="relu",
    solver="adam",
    alpha=0.0001,
    learning_rate_init=0.001,
    max_iter=2000,
    random_state=42,
) -> MLPClassifier:
    """
    Build a scikit-learn MLP classifier.

    Args:
        hidden_layer_sizes: Tuple with the number of neurons in each hidden layer
        activation: Activation function ('identity', 'logistic', 'tanh', 'relu')
        solver: Solver for weight optimization ('lbfgs', 'sgd', 'adam')
        alpha: L2 regularization parameter
        learning_rate_init: Initial learning rate for 'sgd' or 'adam' solvers
        max_iter: Maximum number of iterations
        random_state: Random seed for reproducibility

    Returns:
        Initialized MLPClassifier
    """
    logger.info(f"Building scikit-learn MLP with {hidden_layer_sizes} hidden layers")

    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        random_state=random_state,
        verbose=False,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
    )

    return model


def train_sklearn_mlp(
    model: MLPClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
) -> MLPClassifier:
    """
    Train a scikit-learn MLP classifier.

    Args:
        model: MLPClassifier to train
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (optional)
        y_val: Validation labels (optional)

    Returns:
        Trained model
    """
    logger.info("Training scikit-learn MLP classifier")

    # Train the model
    model.fit(X_train, y_train)

    # Calculate metrics
    y_pred = model.predict(X_train)
    y_pred_proba = model.predict_proba(X_train)[:, 1]

    metrics = {
        "train_accuracy": accuracy_score(y_train, y_pred),
        "train_precision": precision_score(y_train, y_pred),
        "train_recall": recall_score(y_train, y_pred),
        "train_f1": f1_score(y_train, y_pred),
        "train_auc": roc_auc_score(y_train, y_pred_proba),
    }

    logger.info(f"Train: Acc={metrics['train_accuracy']:.4f}, AUC={metrics['train_auc']:.4f}")

    # Validation metrics if validation data is provided
    if X_val is not None and y_val is not None:
        y_val_pred = model.predict(X_val)
        y_val_pred_proba = model.predict_proba(X_val)[:, 1]

        val_metrics = {
            "val_accuracy": accuracy_score(y_val, y_val_pred),
            "val_precision": precision_score(y_val, y_val_pred),
            "val_recall": recall_score(y_val, y_val_pred),
            "val_f1": f1_score(y_val, y_val_pred),
            "val_auc": roc_auc_score(y_val, y_val_pred_proba),
        }

        logger.info(f"Val: Acc={val_metrics['val_accuracy']:.4f}, AUC={val_metrics['val_auc']:.4f}")

    return model


def build_keras_mlp(
    input_dim: int,
    architecture: List[Dict[str, Any]],
    learning_rate: float = 0.001,
    metrics: List[str] = ["accuracy"],
) -> keras.Model:
    """
    Build a Keras MLP model with custom architecture.

    Args:
        input_dim: Dimensionality of the input features
        architecture: List of dictionaries, each describing a hidden layer
            Each dict should have: units, activation, dropout, l2_regularization
        learning_rate: Learning rate for the Adam optimizer
        metrics: List of metrics to track during training

    Returns:
        Compiled Keras model
    """
    logger.info(f"Building Keras MLP with {len(architecture)} hidden layers")

    # Define input layer
    inputs = keras.Input(shape=(input_dim,), name="input")
    x = inputs

    # Add hidden layers based on architecture
    for i, layer_config in enumerate(architecture):
        # Extract layer parameters
        units = layer_config["units"]
        activation = layer_config["activation"]
        dropout_rate = layer_config["dropout"]
        l2_reg = layer_config["l2_regularization"]

        # Replace 'leaky_relu' string with the actual function
        if activation == "leaky_relu":
            act_function = layers.LeakyReLU(alpha=0.1)
        else:
            act_function = activation

        # Add Dense layer with L2 regularization
        x = layers.Dense(
            units=units, kernel_regularizer=regularizers.l2(l2_reg), name=f"dense_{i}"
        )(x)

        # Add activation (as a separate layer if it's leaky_relu)
        if activation == "leaky_relu":
            x = act_function(x)

        # Add dropout layer
        if dropout_rate > 0:
            x = layers.Dropout(rate=dropout_rate, name=f"dropout_{i}")(x)

    # Output layer
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    # Create and compile model
    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=metrics,
    )

    # Display model summary
    model.summary(print_fn=logger.info)

    return model


def train_keras_mlp(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 32,
    epochs: int = 100,
    early_stopping_patience: int = 10,
    reduce_lr_patience: int = 5,
    model_path: str = None,
) -> Tuple[keras.Model, Dict[str, List]]:
    """
    Train a Keras MLP model.

    Args:
        model: Compiled Keras model
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        batch_size: Batch size for training
        epochs: Maximum number of epochs
        early_stopping_patience: Patience for early stopping
        reduce_lr_patience: Patience for learning rate reduction
        model_path: Path to save the best model (optional)

    Returns:
        Tuple of (trained model, training history)
    """
    logger.info(f"Training Keras MLP for up to {epochs} epochs (batch size: {batch_size})")

    # Callbacks
    callbacks_list = []

    # Early stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True
    )
    callbacks_list.append(early_stopping)

    # Reduce learning rate on plateau
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=reduce_lr_patience,
        min_lr=1e-6,
        verbose=1,
    )
    callbacks_list.append(reduce_lr)

    # Model checkpoint if path is provided
    if model_path is not None:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model_checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=model_path, monitor="val_loss", save_best_only=True
        )
        callbacks_list.append(model_checkpoint)

    # Train the model
    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks_list,
        verbose=1,
    )

    # Log final performance
    val_loss = min(history.history["val_loss"])
    val_metrics = {}

    if "val_accuracy" in history.history:
        val_metrics["accuracy"] = max(history.history["val_accuracy"])

    if "val_auc" in history.history:
        val_metrics["auc"] = max(history.history["val_auc"])

    logger.info(f"Training complete. Best val_loss: {val_loss:.4f}")
    for metric_name, value in val_metrics.items():
        logger.info(f"Best val_{metric_name}: {value:.4f}")

    return model, history


def plot_training_history(history: Dict[str, List], save_path: str = None) -> plt.Figure:
    """
    Plot the training history of a Keras model.

    Args:
        history: Training history from keras model.fit()
        save_path: Path to save the figure (optional)

    Returns:
        Matplotlib figure
    """
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot loss
    ax1.plot(history.history["loss"], label="Training Loss")
    if "val_loss" in history.history:
        ax1.plot(history.history["val_loss"], label="Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True)

    # Plot accuracy
    if "accuracy" in history.history:
        ax2.plot(history.history["accuracy"], label="Training Accuracy")
    elif "acc" in history.history:
        ax2.plot(history.history["acc"], label="Training Accuracy")

    if "val_accuracy" in history.history:
        ax2.plot(history.history["val_accuracy"], label="Validation Accuracy")
    elif "val_acc" in history.history:
        ax2.plot(history.history["val_acc"], label="Validation Accuracy")

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training and Validation Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    # Save figure if path is provided
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Training history plot saved to {save_path}")

    return fig


def evaluate_sklearn_mlp(
    model: MLPClassifier, X_test: np.ndarray, y_test: np.ndarray
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Evaluate a trained scikit-learn MLP model on test data.

    Args:
        model: Trained scikit-learn MLPClassifier
        X_test: Test features
        y_test: Test labels

    Returns:
        Tuple of (metrics, predictions, prediction probabilities)
    """
    logger.info("Evaluating scikit-learn MLP model on test data")

    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
    }

    # Log metrics
    logger.info("Test metrics for scikit-learn MLP:")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
    logger.info(f"  ROC AUC:   {metrics['roc_auc']:.4f}")

    return metrics, y_pred, y_pred_proba


def evaluate_keras_mlp(
    model: keras.Model, X_test: np.ndarray, y_test: np.ndarray
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Evaluate a trained Keras MLP model on test data.

    Args:
        model: Trained Keras model
        X_test: Test features
        y_test: Test labels

    Returns:
        Tuple of (metrics, predictions, prediction probabilities)
    """
    logger.info("Evaluating Keras MLP model on test data")

    # Get predictions and convert to 1D array properly to avoid TensorFlow warnings
    pred_raw = model.predict(X_test)
    y_pred_proba = np.reshape(pred_raw, -1)  # Safer than flatten() or ravel()
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
    }

    # Log metrics
    logger.info("Test metrics for Keras MLP:")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
    logger.info(f"  ROC AUC:   {metrics['roc_auc']:.4f}")

    return metrics, y_pred, y_pred_proba


def combine_predictions(
    pred_proba1: np.ndarray, pred_proba2: np.ndarray, method: str = "mean"
) -> np.ndarray:
    """
    Combine prediction probabilities from multiple models.

    Args:
        pred_proba1: Prediction probabilities from first model
        pred_proba2: Prediction probabilities from second model
        method: Combination method ('mean', 'max', 'min', 'product', 'weighted')

    Returns:
        Combined prediction probabilities
    """
    if method == "mean":
        return (pred_proba1 + pred_proba2) / 2
    elif method == "max":
        return np.maximum(pred_proba1, pred_proba2)
    elif method == "min":
        return np.minimum(pred_proba1, pred_proba2)
    elif method == "product":
        return np.sqrt(pred_proba1 * pred_proba2)  # Geometric mean
    elif method == "weighted":
        # Example with fixed weights - could be parameters
        return 0.4 * pred_proba1 + 0.6 * pred_proba2
    else:
        raise ValueError(f"Unknown combination method: {method}")


# This function is intentionally complex due to the comprehensive interpretation logic
# noqa: C901
def interpret_prediction(  # noqa: C901
    model=None,
    patient_data: Dict[str, float] = None,
    feature_names: List[str] = None,
    probability: float = None,
) -> str:
    """
    Interpret a model prediction for a single patient.

    Args:
        model: Trained model (optional)
        patient_data: Dictionary of patient features
        feature_names: List of feature names (optional)
        probability: Prediction probability (used when model is None)

    Returns:
        String interpretation of the prediction
    """
    # Set up logger
    logger = logging.getLogger(__name__)

    # Handle edge cases
    if patient_data is None:
        logger.warning("No patient data provided for interpretation")
        return "No patient data provided for interpretation."

    # If model is provided, use it to get prediction
    if model is not None and feature_names is not None:
        try:
            # Convert patient data to feature array
            X = np.array([[patient_data[f] for f in feature_names]])

            # Make prediction
            if isinstance(model, keras.Model):
                probability = model.predict(X)[0][0]
            else:
                probability = model.predict_proba(X)[0][1]
        except Exception as e:
            logger.error(f"Error predicting with model: {e}")
            # Use default probability if prediction fails
            if probability is None:
                probability = 0.5

    # Handle missing probability
    if probability is None:
        logger.warning("Missing probability value, using default of 0.5")
        probability = 0.5  # Use a default value instead of returning early

    # Start building the interpretation
    interpretation = []

    # Ensure probability is a float (important for comparison)
    try:
        probability_float = float(probability)
    except (TypeError, ValueError):
        logger.warning(f"Invalid probability value: {probability}, using default of 0.5")
        probability_float = 0.5  # Fallback to default

    # High vs low risk determination
    if probability_float > 0.5:
        interpretation.append(
            f"HIGH RISK PREDICTION: {probability_float:.1%} probability of heart disease"
        )

        # Identify risk factors
        risk_factors = []

        # Check common risk factors - with safe access to dict keys
        if patient_data.get("age", 0) > 55:
            risk_factors.append("Advanced age (over 55)")

        if patient_data.get("sex", 0) == 1:
            risk_factors.append("Male over 45")

        if patient_data.get("trestbps", 0) > 140:
            risk_factors.append("Elevated resting blood pressure")

        if patient_data.get("chol", 0) > 240:
            risk_factors.append("High cholesterol")

        if patient_data.get("fbs", 0) == 1:
            risk_factors.append("Fasting blood sugar > 120 mg/dl")

        if patient_data.get("thalach", 999) < 150:  # Use high default to avoid false positive
            risk_factors.append("Reduced maximum heart rate")

        if patient_data.get("exang", 0) == 1:
            risk_factors.append("Exercise-induced angina")

        # Add risk factors to interpretation
        if risk_factors:
            interpretation.append("\nKey risk factors identified:")
            for factor in risk_factors:
                interpretation.append(f"- {factor}")
        else:
            interpretation.append("\nNo specific risk factors identified in the provided data.")

        # Add recommendations
        interpretation.append("\nRecommendations:")
        interpretation.append("- Consult with a cardiologist")
        interpretation.append("- Consider stress test or other cardiac evaluations")
        interpretation.append("- Review medication and lifestyle modifications")

    else:
        interpretation.append(
            f"LOW RISK PREDICTION: {probability_float:.1%} probability of heart disease"
        )

        # For low risk, do a quick check if there are any risk factors - with safe access
        has_risk_factors = (
            patient_data.get("age", 0) > 45
            or patient_data.get("sex", 0) == 1
            or patient_data.get("trestbps", 0) > 130
            or patient_data.get("chol", 0) > 200
            or patient_data.get("fbs", 0) == 1
            or patient_data.get("thalach", 999) < 150
            or patient_data.get("exang", 0) == 1  # Use high default to avoid false positive
        )

        if has_risk_factors:
            interpretation.append("\nSome risk factors present, but overall risk is low.")
            interpretation.append("\nRecommendations:")
            interpretation.append("- Continue regular check-ups")
            interpretation.append("- Maintain heart-healthy lifestyle")
        else:
            interpretation.append("\nNo major risk factors identified.")
            interpretation.append("\nRecommendations:")
            interpretation.append("- Continue regular check-ups")
            interpretation.append("- Maintain heart-healthy lifestyle")

    # Return the complete interpretation as a string
    return "\n".join(interpretation)
