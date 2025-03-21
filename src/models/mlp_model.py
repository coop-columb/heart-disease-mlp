"""
MLP model implementations for the Heart Disease Prediction project.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import os
import joblib
from typing import Dict, List, Any, Union, Optional, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def build_sklearn_mlp(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    learning_rate_init=0.001,
    max_iter=2000,
    random_state=42
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
        n_iter_no_change=10
    )
    
    return model


def train_sklearn_mlp(
    model: MLPClassifier, 
    X_train: np.ndarray, 
    y_train: np.ndarray,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None
) -> Tuple[MLPClassifier, Dict[str, float]]:
    """
    Train a scikit-learn MLP classifier.
    
    Args:
        model: MLPClassifier to train
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        
    Returns:
        Tuple of (trained model, metrics dict)
    """
    logger.info("Training scikit-learn MLP classifier")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Calculate metrics
    y_pred = model.predict(X_train)
    y_pred_proba = model.predict_proba(X_train)[:, 1]
    
    metrics = {
        'train_accuracy': accuracy_score(y_train, y_pred),
        'train_precision': precision_score(y_train, y_pred),
        'train_recall': recall_score(y_train, y_pred),
        'train_f1': f1_score(y_train, y_pred),
        'train_auc': roc_auc_score(y_train, y_pred_proba)
    }
    
    logger.info(f"Training metrics: Accuracy={metrics['train_accuracy']:.4f}, AUC={metrics['train_auc']:.4f}")
    
    # Validation metrics if validation data is provided
    if X_val is not None and y_val is not None:
        y_val_pred = model.predict(X_val)
        y_val_pred_proba = model.predict_proba(X_val)[:, 1]
        
        metrics.update({
            'val_accuracy': accuracy_score(y_val, y_val_pred),
            'val_precision': precision_score(y_val, y_val_pred),
            'val_recall': recall_score(y_val, y_val_pred),
            'val_f1': f1_score(y_val, y_val_pred),
            'val_auc': roc_auc_score(y_val, y_val_pred_proba)
        })
        
        logger.info(f"Validation metrics: Accuracy={metrics['val_accuracy']:.4f}, AUC={metrics['val_auc']:.4f}")
    
    return model, metrics


def build_keras_mlp(
    input_dim: int,
    architecture: List[Dict[str, Any]],
    learning_rate: float = 0.001,
    metrics: List[str] = ['accuracy']
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
    inputs = keras.Input(shape=(input_dim,), name='input')
    x = inputs
    
    # Add hidden layers based on architecture
    for i, layer_config in enumerate(architecture):
        # Extract layer parameters
        units = layer_config['units']
        activation = layer_config['activation']
        dropout_rate = layer_config['dropout']
        l2_reg = layer_config['l2_regularization']
        
        # Replace 'leaky_relu' string with the actual function
        if activation == 'leaky_relu':
            act_function = layers.LeakyReLU(alpha=0.1)
        else:
            act_function = activation
        
        # Add Dense layer with L2 regularization
        x = layers.Dense(
            units=units,
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f'dense_{i}'
        )(x)
        
        # Add activation (as a separate layer if it's leaky_relu)
        if activation == 'leaky_relu':
            x = act_function(x)
        
        # Add dropout layer
        if dropout_rate > 0:
            x = layers.Dropout(rate=dropout_rate, name=f'dropout_{i}')(x)
    
    # Output layer
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    # Create and compile model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=metrics
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
    patience: int = 10,
    model_path: str = None
) -> Tuple[keras.Model, Dict[str, Any], Dict[str, List]]:
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
        patience: Patience for early stopping
        model_path: Path to save the best model (optional)
        
    Returns:
        Tuple of (trained model, metrics dict, training history)
    """
    logger.info(f"Training Keras MLP for up to {epochs} epochs (batch size: {batch_size})")
    
    # Callbacks
    callbacks_list = []
    
    # Early stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True
    )
    callbacks_list.append(early_stopping)
    
    # Model checkpoint if path is provided
    if model_path is not None:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model_checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor='val_loss',
            save_best_only=True
        )
        callbacks_list.append(model_checkpoint)
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks_list,
        verbose=1
    )
    
    # Get metrics from the last epoch
    metrics = {}
    for metric_name, value in history.history.items():
        metrics[metric_name] = value[-1]
    
    # Calculate additional metrics on validation set
    y_val_pred = (model.predict(X_val) > 0.5).astype(int).flatten()
    y_val_pred_proba = model.predict(X_val).flatten()
    
    metrics.update({
        'val_accuracy': accuracy_score(y_val, y_val_pred),
        'val_precision': precision_score(y_val, y_val_pred),
        'val_recall': recall_score(y_val, y_val_pred),
        'val_f1': f1_score(y_val, y_val_pred),
        'val_auc': roc_auc_score(y_val, y_val_pred_proba)
    })
    
    logger.info(f"Final validation metrics: Accuracy={metrics['val_accuracy']:.4f}, AUC={metrics['val_auc']:.4f}")
    
    return model, metrics, history.history


def plot_training_history(
    history: Dict[str, List], 
    save_path: str = None
) -> plt.Figure:
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
    ax1.plot(history['loss'], label='Training Loss')
    if 'val_loss' in history:
        ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    if 'accuracy' in history:
        ax2.plot(history['accuracy'], label='Training Accuracy')
    elif 'acc' in history:
        ax2.plot(history['acc'], label='Training Accuracy')
        
    if 'val_accuracy' in history:
        ax2.plot(history['val_accuracy'], label='Validation Accuracy')
    elif 'val_acc' in history:
        ax2.plot(history['val_acc'], label='Validation Accuracy')
        
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history plot saved to {save_path}")
    
    return fig


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained scikit-learn or Keras model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating model on test data")
    
    # Check if model is a Keras model
    is_keras = isinstance(model, keras.Model)
    
    if is_keras:
        # Keras model predictions
        y_pred_proba = model.predict(X_test).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
    else:
        # Scikit-learn model predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    # Log metrics
    logger.info(f"Test metrics:")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
    logger.info(f"  ROC AUC:   {metrics['auc']:.4f}")
    
    return metrics