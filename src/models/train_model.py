"""
Script to train heart disease prediction models.
"""
import argparse
import logging
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any

import tensorflow as tf
from tensorflow import keras

from src.utils import load_config
from src.models.mlp_model import (
    build_sklearn_mlp, train_sklearn_mlp, evaluate_sklearn_mlp,
    build_keras_mlp, train_keras_mlp, evaluate_keras_mlp,
    combine_predictions
)
from src.models.hyperparameter_tuning import tune_sklearn_mlp, tune_keras_mlp
from src.visualization.visualize import (
    plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve,
    plot_feature_importance
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_processed_data(processed_data_path):
    """
    Load processed data for model training.
    
    Args:
        processed_data_path: Path to processed data file
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    logger.info(f"Loading processed data from {processed_data_path}")
    
    try:
        # Load data
        with np.load(processed_data_path) as data:
            X_train = data['X_train']
            X_val = data['X_val']
            X_test = data['X_test']
            y_train = data['y_train']
            y_val = data['y_val']
            y_test = data['y_test']
        
        logger.info(f"Loaded data shapes: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    except Exception as e:
        logger.error(f"Error loading processed data: {e}")
        raise


def train_models(X_train, X_val, X_test, y_train, y_val, y_test, config, model_dir="models"):
    """
    Train and evaluate heart disease prediction models.
    
    Args:
        X_train: Training features
        X_val: Validation features
        X_test: Test features
        y_train: Training targets
        y_val: Validation targets
        y_test: Test targets
        config: Configuration dictionary
        model_dir: Directory to save models
        
    Returns:
        Dictionary of trained models and evaluation results
    """
    logger.info("Starting model training and evaluation")
    
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Extract model parameters from config
    sklearn_mlp_params = config['model']['mlp']
    keras_mlp_params = config['model']['advanced_mlp']
    
    # Initialize results dictionary
    results = {}
    
    # Train scikit-learn MLP model
    logger.info("Training scikit-learn MLP model")
    sklearn_mlp = build_sklearn_mlp(
        hidden_layer_sizes=tuple(sklearn_mlp_params['hidden_layer_sizes']),
        activation=sklearn_mlp_params['activation'],
        solver=sklearn_mlp_params['solver'],
        alpha=sklearn_mlp_params['alpha'],
        learning_rate_init=sklearn_mlp_params['learning_rate_init'],
        max_iter=sklearn_mlp_params['max_iter'],
        random_state=sklearn_mlp_params['random_state']
    )
    
    sklearn_mlp = train_sklearn_mlp(sklearn_mlp, X_train, y_train, X_val, y_val)
    
    # Evaluate scikit-learn MLP
    sklearn_metrics, sklearn_y_pred, sklearn_y_pred_proba = evaluate_sklearn_mlp(sklearn_mlp, X_test, y_test)
    
    # Save scikit-learn model
    joblib.dump(sklearn_mlp, os.path.join(model_dir, "sklearn_mlp_model.joblib"))
    logger.info(f"Saved scikit-learn MLP model to {os.path.join(model_dir, 'sklearn_mlp_model.joblib')}")
    
    # Store results
    results['sklearn_mlp'] = {
        'model': sklearn_mlp,
        'metrics': sklearn_metrics,
        'predictions': sklearn_y_pred,
        'probabilities': sklearn_y_pred_proba
    }
    
    # Train TensorFlow/Keras MLP model
    logger.info("Training Keras MLP model")
    input_dim = X_train.shape[1]
    
    keras_mlp = build_keras_mlp(
        input_dim=input_dim,
        architecture=keras_mlp_params['architecture'],
        learning_rate=keras_mlp_params['learning_rate'],
        metrics=['accuracy', keras.metrics.AUC(), keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    keras_mlp, history = train_keras_mlp(
        keras_mlp,
        X_train, y_train,
        X_val, y_val,
        batch_size=keras_mlp_params['batch_size'],
        epochs=keras_mlp_params['epochs'],
        early_stopping_patience=keras_mlp_params['early_stopping_patience'],
        reduce_lr_patience=keras_mlp_params['reduce_lr_patience']
    )
    
    # Evaluate Keras MLP
    keras_metrics, keras_y_pred, keras_y_pred_proba = evaluate_keras_mlp(keras_mlp, X_test, y_test)
    
    # Save Keras model
    keras_mlp.save(os.path.join(model_dir, "keras_mlp_model.h5"))
    logger.info(f"Saved Keras MLP model to {os.path.join(model_dir, 'keras_mlp_model.h5')}")
    
    # Store results
    results['keras_mlp'] = {
        'model': keras_mlp,
        'history': history,
        'metrics': keras_metrics,
        'predictions': keras_y_pred,
        'probabilities': keras_y_pred_proba
    }
    
    # Ensemble prediction (simple average)
    logger.info("Creating ensemble prediction")
    ensemble_y_pred_proba = combine_predictions(sklearn_y_pred_proba, keras_y_pred_proba, method='mean')
    ensemble_y_pred = (ensemble_y_pred_proba >= 0.5).astype(int)
    
    # Calculate ensemble metrics
    ensemble_metrics = {
        'accuracy': accuracy_score(y_test, ensemble_y_pred),
        'precision': precision_score(y_test, ensemble_y_pred),
        'recall': recall_score(y_test, ensemble_y_pred),
        'f1': f1_score(y_test, ensemble_y_pred),
        'roc_auc': roc_auc_score(y_test, ensemble_y_pred_proba)
    }
    
    # Log ensemble results
    logger.info("Ensemble Model Evaluation:")
    for metric, value in ensemble_metrics.items():
        logger.info(f"{metric.capitalize()}: {value:.4f}")
    
    # Store ensemble results
    results['ensemble'] = {
        'metrics': ensemble_metrics,
        'predictions': ensemble_y_pred,
        'probabilities': ensemble_y_pred_proba
    }
    
    # Save evaluation results
    evaluation_results = {
        'sklearn_mlp': sklearn_metrics,
        'keras_mlp': keras_metrics,
        'ensemble': ensemble_metrics
    }
    
    joblib.dump(evaluation_results, os.path.join(model_dir, "evaluation_results.joblib"))
    logger.info(f"Saved evaluation results to {os.path.join(model_dir, 'evaluation_results.joblib')}")
    
    # Create evaluation visualizations
    create_evaluation_visualizations(y_test, results, config)
    
    return results


def create_evaluation_visualizations(y_test, results, config):
    """
    Create and save evaluation visualizations.
    
    Args:
        y_test: Test targets
        results: Dictionary of model results
        config: Configuration dictionary
    """
    logger.info("Creating evaluation visualizations")
    
    # Create figures directory if it doesn't exist
    os.makedirs("reports/figures", exist_ok=True)
    
    # Create confusion matrices
    for model_name, model_results in results.items():
        # Skip if model doesn't have predictions
        if 'predictions' not in model_results:
            continue
        
        # Confusion matrix
        plt_cm = plot_confusion_matrix(
            y_test, 
            model_results['predictions'],
            title=f"Confusion Matrix - {model_name}"
        )
        plt_cm.savefig(f"reports/figures/confusion_matrix_{model_name}.png")
        plt_cm.close()
        
        # ROC curve
        plt_roc = plot_roc_curve(
            y_test, 
            model_results['probabilities'],
            title=f"ROC Curve - {model_name}"
        )
        plt_roc.savefig(f"reports/figures/roc_curve_{model_name}.png")
        plt_roc.close()
        
        # Precision-Recall curve
        plt_pr = plot_precision_recall_curve(
            y_test, 
            model_results['probabilities'],
            title=f"Precision-Recall Curve - {model_name}"
        )
        plt_pr.savefig(f"reports/figures/pr_curve_{model_name}.png")
        plt_pr.close()
    
    # Learning curves for Keras model
    if 'keras_mlp' in results and 'history' in results['keras_mlp']:
        history = results['keras_mlp']['history']
        
        # Plot accuracy
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')
        plt.savefig("reports/figures/keras_accuracy.png")
        plt.close()
        
        # Plot loss
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.savefig("reports/figures/keras_loss.png")
        plt.close()


def main(config_path="config/config.yaml", processed_data_path=None, model_dir="models", tune=False):
    """
    Main function to train heart disease prediction models.
    
    Args:
        config_path: Path to configuration file
        processed_data_path: Path to processed data (overrides config if provided)
        model_dir: Directory to save models
        tune: Whether to perform hyperparameter tuning
    """
    # Load configuration
    config = load_config(config_path)
    
    # Override paths if provided
    if processed_data_path is None:
        processed_data_path = os.path.join(
            os.path.dirname(config['data']['processed_data_path']),
            'processed_data.npz'
        )
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data(processed_data_path)
    
    # Perform hyperparameter tuning if requested
    if tune:
        logger.info("Performing hyperparameter tuning")
        
        # Tune scikit-learn MLP
        sklearn_best_params, sklearn_study = tune_sklearn_mlp(
            X_train, y_train, n_trials=50, cv=5, random_state=42
        )
        
        # Tune Keras MLP
        keras_best_params, keras_study = tune_keras_mlp(
            X_train, y_train, X_val, y_val, n_trials=30, random_state=42
        )
        
        # Update config with best parameters
        # This is a simplified version; in a real setup, you might want to update the config file
        config['model']['mlp']['hidden_layer_sizes'] = sklearn_best_params['hidden_layer_sizes']
        config['model']['mlp']['activation'] = sklearn_best_params['activation']
        config['model']['mlp']['solver'] = sklearn_best_params['solver']
        config['model']['mlp']['alpha'] = sklearn_best_params['alpha']
        config['model']['mlp']['learning_rate_init'] = sklearn_best_params['learning_rate_init']
        
        config['model']['advanced_mlp']['architecture'] = keras_best_params['architecture']
        config['model']['advanced_mlp']['learning_rate'] = keras_best_params['learning_rate']
        config['model']['advanced_mlp']['batch_size'] = keras_best_params['batch_size']
    
    # Train models
    results = train_models(X_train, X_val, X_test, y_train, y_val, y_test, config, model_dir)
    
    logger.info("Model training and evaluation complete")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train heart disease prediction models")
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--processed-data-path", type=str, default=None,
        help="Path to processed data (overrides config if provided)"
    )
    parser.add_argument(
        "--model-dir", type=str, default="models",
        help="Directory to save models"
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Perform hyperparameter tuning"
    )
    
    args = parser.parse_args()
    main(args.config, args.processed_data_path, args.model_dir, args.tune)