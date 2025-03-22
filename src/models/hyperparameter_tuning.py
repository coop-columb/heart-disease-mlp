"""
Hyperparameter tuning for MLP models using Optuna.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import os
import joblib
from typing import Dict, List, Any, Union, Optional, Tuple

import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.models.mlp_model import build_sklearn_mlp, build_keras_mlp, train_keras_mlp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def tune_sklearn_mlp(X_train, y_train, n_trials=100, cv=5, random_state=42):
    """
    Tune scikit-learn MLP hyperparameters using Optuna.
    
    Args:
        X_train: Training features
        y_train: Training targets
        n_trials: Number of optimization trials
        cv: Number of cross-validation folds
        random_state: Random seed for reproducibility
        
    Returns:
        Best parameters and study object
    """
    logger.info(f"Starting scikit-learn MLP hyperparameter tuning with {n_trials} trials")
    
    # Define objective function
    def objective(trial):
        # Define hyperparameters to tune
        hidden_layer_sizes = trial.suggest_categorical(
            'hidden_layer_sizes',
            [
                (50,), (100,), (200,),
                (50, 25), (100, 50), (200, 100),
                (100, 50, 25), (200, 100, 50)
            ]
        )
        
        activation = trial.suggest_categorical(
            'activation', ['relu', 'tanh', 'logistic']
        )
        
        solver = trial.suggest_categorical(
            'solver', ['adam', 'sgd', 'lbfgs']
        )
        
        alpha = trial.suggest_float(
            'alpha', 1e-5, 1e-2, log=True
        )
        
        learning_rate_init = trial.suggest_float(
            'learning_rate_init', 1e-4, 1e-1, log=True
        )
        
        # Create MLP model
        model = build_sklearn_mlp(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=1000,
            random_state=random_state
        )
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train, 
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state),
            scoring='roc_auc',
            n_jobs=-1
        )
        
        return cv_scores.mean()
    
    # Create study
    study = optuna.create_study(direction='maximize', study_name='sklearn_mlp_tuning')
    
    # Optimize
    study.optimize(objective, n_trials=n_trials)
    
    # Log best parameters
    logger.info(f"Best ROC AUC: {study.best_value:.4f}")
    logger.info(f"Best parameters: {study.best_params}")
    
    # Save study
    os.makedirs('models/optuna', exist_ok=True)
    joblib.dump(study, 'models/optuna/sklearn_mlp_study.pkl')
    
    return study.best_params, study


def tune_keras_mlp(X_train, y_train, X_val, y_val, n_trials=50, random_state=42):
    """
    Tune Keras MLP hyperparameters using Optuna.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        n_trials: Number of optimization trials
        random_state: Random seed for reproducibility
        
    Returns:
        Best parameters and study object
    """
    logger.info(f"Starting Keras MLP hyperparameter tuning with {n_trials} trials")
    
    # Set random seeds for reproducibility
    np.random.seed(random_state)
    tf.random.set_seed(random_state)
    
    input_dim = X_train.shape[1]
    
    # Define objective function
    def objective(trial):
        # Number of layers
        n_layers = trial.suggest_int('n_layers', 1, 3)
        
        # Layer architecture
        architecture = []
        for i in range(n_layers):
            units = trial.suggest_int(f'units_layer{i}', 16, 256, log=True)
            
            # Activation function
            activation = trial.suggest_categorical(
                f'activation_layer{i}', ['relu', 'leaky_relu', 'tanh']
            )
            
            # Dropout rate
            dropout = trial.suggest_float(f'dropout_layer{i}', 0.0, 0.5)
            
            # L2 regularization
            l2_reg = trial.suggest_float(f'l2_reg_layer{i}', 1e-5, 1e-2, log=True)
            
            # Add layer configuration
            architecture.append({
                'units': units,
                'activation': activation,
                'dropout': dropout,
                'l2_regularization': l2_reg
            })
        
        # Learning rate
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        
        # Batch size
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        
        # Create and compile model
        model = build_keras_mlp(
            input_dim=input_dim,
            architecture=architecture,
            learning_rate=learning_rate,
            metrics=['accuracy', keras.metrics.AUC()]
        )
        
        # Train model
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Evaluate model
        val_loss = min(history.history['val_loss'])
        val_auc = max(history.history['val_auc'])
        
        return val_auc
    
    # Create study
    study = optuna.create_study(direction='maximize', study_name='keras_mlp_tuning')
    
    # Optimize
    study.optimize(objective, n_trials=n_trials)
    
    # Log best parameters
    logger.info(f"Best validation AUC: {study.best_value:.4f}")
    logger.info(f"Best parameters: {study.best_params}")
    
    # Save study
    os.makedirs('models/optuna', exist_ok=True)
    joblib.dump(study, 'models/optuna/keras_mlp_study.pkl')
    
    # Generate optimization plots
    os.makedirs('reports/figures', exist_ok=True)
    
    # Plot optimization history
    fig = plot_optimization_history(study)
    fig.write_image('reports/figures/keras_optimization_history.png')
    
    # Plot parameter importances
    fig = plot_param_importances(study)
    fig.write_image('reports/figures/keras_param_importances.png')
    
    # Construct best architecture from study results
    best_params = study.best_params
    n_layers = best_params['n_layers']
    
    best_architecture = []
    for i in range(n_layers):
        layer = {
            'units': best_params[f'units_layer{i}'],
            'activation': best_params[f'activation_layer{i}'],
            'dropout': best_params[f'dropout_layer{i}'],
            'l2_regularization': best_params[f'l2_reg_layer{i}']
        }
        best_architecture.append(layer)
    
    # Return best architecture and other parameters
    best_hyperparams = {
        'architecture': best_architecture,
        'learning_rate': best_params['learning_rate'],
        'batch_size': best_params['batch_size']
    }
    
    return best_hyperparams, study