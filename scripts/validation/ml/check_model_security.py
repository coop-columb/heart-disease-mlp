#!/usr/bin/env python3
"""
Check ML model for security vulnerabilities.

This script:
1. Tests for adversarial attack vulnerability
2. Checks for membership inference attacks
3. Validates robustness to input perturbations
4. Examines model attack surface
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

try:
    # Optional imports for advanced attacks
    from heart_disease.models.mlp_model import load_model
except ImportError:
    logger.warning("Could not import heart_disease.models module")

def generate_adversarial_examples(
    model: Any,
    X: np.ndarray,
    epsilon: float = 0.1,
    norm: str = "l_inf"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate adversarial examples using Fast Gradient Sign Method (FGSM).
    
    Args:
        model: Model to attack
        X: Input features
        epsilon: Attack strength
        norm: Norm to use (l_inf, l2, l1)
        
    Returns:
        Tuple of (perturbed_examples, perturbations)
    """
    try:
        # Save original predictions for later comparison
        original_predictions = model.predict(X)
        
        # Check if model has a gradient method directly
        if hasattr(model, "compute_gradients"):
            # Use model's gradient function
            gradients = model.compute_gradients(X)
        elif hasattr(model, "predict_proba"):
            # Numerical gradient approximation for black-box models
            gradients = np.zeros_like(X)
            delta = 1e-6
            
            for i in range(X.shape[1]):
                X_plus = X.copy()
                X_plus[:, i] += delta
                X_minus = X.copy()
                X_minus[:, i] -= delta
                
                pred_plus = model.predict_proba(X_plus)[:, 1]
                pred_minus = model.predict_proba(X_minus)[:, 1]
                
                gradients[:, i] = (pred_plus - pred_minus) / (2 * delta)
        else:
            # Fallback if we can't compute gradients
            logger.warning("Unable to compute gradients for model - using random perturbations")
            gradients = np.random.normal(0, 1, X.shape)
        
        # Create perturbation based on the norm
        if norm == "l_inf":
            # L-infinity norm (maximum absolute value)
            perturbation = epsilon * np.sign(gradients)
        elif norm == "l2":
            # L2 norm (Euclidean distance)
            l2_norms = np.sqrt(np.sum(gradients**2, axis=1, keepdims=True))
            l2_norms = np.maximum(l2_norms, 1e-12)  # Avoid division by zero
            normalized_gradients = gradients / l2_norms
            perturbation = epsilon * normalized_gradients
        elif norm == "l1":
            # L1 norm (sum of absolute values)
            l1_norms = np.sum(np.abs(gradients), axis=1, keepdims=True)
            l1_norms = np.maximum(l1_norms, 1e-12)  # Avoid division by zero
            normalized_gradients = gradients / l1_norms
            perturbation = epsilon * normalized_gradients
        else:
            raise ValueError(f"Unknown norm: {norm}")
        
        # Create adversarial examples
        adversarial_examples = X + perturbation
        
        # Ensure the perturbed examples remain in valid range
        # Clip to original min/max per feature
        feature_min = np.min(X, axis=0)
        feature_max = np.max(X, axis=0)
        
        adversarial_examples = np.clip(
            adversarial_examples, 
            feature_min, 
            feature_max
        )
        
        return adversarial_examples, perturbation
        
    except Exception as e:
        logger.error(f"Error generating adversarial examples: {e}")
        raise

def test_adversarial_robustness(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    epsilon_range: List[float] = [0.01, 0.05, 0.1, 0.2, 0.3],
    norm: str = "l_inf"
) -> Tuple[bool, Dict]:
    """
    Test model robustness against adversarial attacks.
    
    Args:
        model: Model to test
        X: Input features
        y: Target labels
        epsilon_range: List of attack strengths to test
        norm: Norm to use for perturbations
        
    Returns:
        Tuple of (passed, details)
    """
    results = {
        "accuracies": {},
        "perturbation_strength": {},
        "max_safe_epsilon": None,
        "issues": []
    }
    
    # Calculate baseline accuracy
    baseline_preds = model.predict(X)
    baseline_accuracy = accuracy_score(y, baseline_preds)
    results["baseline_accuracy"] = float(baseline_accuracy)
    
    # Test increasing attack strengths
    passed = True
    for epsilon in epsilon_range:
        try:
            # Generate adversarial examples
            X_adv, perturbation = generate_adversarial_examples(
                model, X, epsilon, norm
            )
            
            # Get predictions on adversarial examples
            adv_preds = model.predict(X_adv)
            adv_accuracy = accuracy_score(y, adv_preds)
            
            # Calculate average perturbation strength
            if norm == "l_inf":
                pert_strength = np.max(np.abs(perturbation))
            elif norm == "l2":
                pert_strength = np.mean(np.sqrt(np.sum(perturbation**2, axis=1)))
            elif norm == "l1":
                pert_strength = np.mean(np.sum(np.abs(perturbation), axis=1))
            
            # Store results
            results["accuracies"][str(epsilon)] = float(adv_accuracy)
            results["perturbation_strength"][str(epsilon)] = float(pert_strength)
            
            # Check if accuracy dropped significantly (more than 30%)
            if adv_accuracy < baseline_accuracy * 0.7:
                if results["max_safe_epsilon"] is None or epsilon < results["max_safe_epsilon"]:
                    results["max_safe_epsilon"] = epsilon
                
                if epsilon <= 0.05:  # Very small perturbations causing issues
                    passed = False
                    results["issues"].append(
                        f"Model is vulnerable to small perturbations (ε={epsilon}, "
                        f"accuracy drop: {baseline_accuracy:.3f} → {adv_accuracy:.3f})"
                    )
            
        except Exception as e:
            logger.warning(f"Error testing epsilon {epsilon}: {e}")
            results["issues"].append(f"Error testing epsilon {epsilon}: {e}")
    
    # If we didn't find a breaking point, that's suspicious (model might be trivial)
    if not results["issues"] and all(acc > baseline_accuracy * 0.9 for acc in results["accuracies"].values()):
        results["issues"].append(
            "Model is suspiciously robust to adversarial examples - check if it's a trivial model"
        )
    
    # Final determination
    passed = len(results["issues"]) == 0
    
    return passed, results

def test_membership_inference(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    threshold: float = 0.65
) -> Tuple[bool, Dict]:
    """
    Test model vulnerability to membership inference attacks.
    
    Args:
        model: Model to test
        X_train: Training data features
        X_test: Test data features
        threshold: AUC threshold for determining vulnerability
        
    Returns:
        Tuple of (passed, details)
    """
    results = {
        "auc_score": None,
        "confidence_gap": None,
        "issues": []
    }
    
    try:
        # Step 1: Calculate prediction confidence for training and test examples
        train_probs = model.predict_proba(X_train)
        test_probs = model.predict_proba(X_test)
        
        # Get confidence scores (max probability)
        train_confidence = np.max(train_probs, axis=1)
        test_confidence = np.max(test_probs, axis=1)
        
        # Calculate average confidence gap
        avg_train_conf = np.mean(train_confidence)
        avg_test_conf = np.mean(test_confidence)
        confidence_gap = avg_train_conf - avg_test_conf
        
        results["confidence_gap"] = float(confidence_gap)
        
        # Step 2: Train a simple membership classifier
        # Create dataset with confidence scores as features
        membership_X = np.concatenate([train_confidence, test_confidence])
        membership_y = np.concatenate([np.ones(len(train_confidence)), np.zeros(len(test_confidence))])
        
        # Shuffle the data
        indices = np.arange(len(membership_X))
        np.random.shuffle(indices)
        membership_X = membership_X[indices]
        membership_y = membership_y[indices]
        
        # Split into train and test for the membership classifier
        X_mem_train, X_mem_test, y_mem_train, y_mem_test = train_test_split(
            membership_X.reshape(-1, 1), membership_y, test_size=0.3, random_state=42
        )
        
        # Train a simple threshold classifier
        from sklearn.linear_model import LogisticRegression
        membership_clf = LogisticRegression(solver='lbfgs')
        membership_clf.fit(X_mem_train, y_mem_train)
        
        # Evaluate the membership classifier
        mem_probs = membership_clf.predict_proba(X_mem_test)[:, 1]
        auc = roc_auc_score(y_mem_test, mem_probs)
        
        results["auc_score"] = float(auc)
        
        # Assess vulnerability
        if auc > threshold:
            results["issues"].append(
                f"Model may be vulnerable to membership inference (AUC: {auc:.3f}, "
                f"threshold: {threshold})"
            )
        
        if confidence_gap > 0.2:
            results["issues"].append(
                f"Large confidence gap between train and test data ({confidence_gap:.3f}), "
                f"indicating potential privacy risk"
            )
        
        # Final determination
        passed = len(results["issues"]) == 0
        
    except Exception as e:
        logger.error(f"Error testing membership inference: {e}")
        results["issues"].append(f"Error testing membership inference: {e}")
        passed = False
    
    return passed, results

def test_model_inversion(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    categorical_features: List[int] = None,
) -> Tuple[bool, Dict]:
    """
    Test model vulnerability to model inversion attacks.
    
    Args:
        model: Model to test
        X: Input features
        y: Target labels
        categorical_features: Indices of categorical features
        
    Returns:
        Tuple of (passed, details)
    """
    results = {
        "feature_importance": {},
        "issues": []
    }
    
    try:
        # Get average feature values
        feature_means = np.mean(X, axis=0)
        
        # Test each feature to see if it can be reconstructed from the others
        for i in range(X.shape[1]):
            # Skip categorical features if specified
            if categorical_features and i in categorical_features:
                continue
            
            # Create a test instance with the feature set to mean
            X_test = np.ones((1, X.shape[1])) * feature_means
            
            # Try different values for the target feature
            min_val = np.min(X[:, i])
            max_val = np.max(X[:, i])
            steps = np.linspace(min_val, max_val, 10)
            
            probs = []
            for val in steps:
                X_test[0, i] = val
                prob = model.predict_proba(X_test)[0, 1]
                probs.append(prob)
            
            # Calculate the maximum change in probability
            max_change = np.max(probs) - np.min(probs)
            results["feature_importance"][i] = float(max_change)
            
            # If a feature causes large changes in prediction, it's sensitive to inversion
            if max_change > 0.5:
                results["issues"].append(
                    f"Feature {i} strongly influences predictions (max change: {max_change:.3f}), "
                    f"making it vulnerable to inversion attacks"
                )
                
    except Exception as e:
        logger.error(f"Error testing model inversion: {e}")
        results["issues"].append(f"Error testing model inversion: {e}")
    
    # Final determination
    passed = len(results["issues"]) == 0
    
    return passed, results

def check_model_security(
    model_path: str,
    data_path: str,
    epsilon: float = 0.1,
    membership_threshold: float = 0.65,
    categorical_features: Optional[List[int]] = None,
    output_path: Optional[str] = None
) -> Tuple[bool, Dict]:
    """
    Run comprehensive model security checks.
    
    Args:
        model_path: Path to the model file
        data_path: Path to the data file (with train/test split)
        epsilon: Perturbation strength for adversarial testing
        membership_threshold: Threshold for membership inference
        categorical_features: List of categorical feature indices
        output_path: Optional path to save results
        
    Returns:
        Tuple of (passed, results)
    """
    try:
        # Load model
        model = load_model(model_path)
        
        # Load data - assuming data file has X_train, X_test, y_train, y_test
        data = np.load(data_path)
        X_train = data['X_train']
        X_test = data['X_test

