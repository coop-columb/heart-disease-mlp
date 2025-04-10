#!/usr/bin/env python3
"""
Validate ML model performance against defined thresholds.

This script:
1. Loads models and test data
2. Evaluates performance metrics
3. Checks against thresholds
4. Reports detailed results
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                           recall_score, roc_auc_score)

from heart_disease.models.mlp_model import load_model, load_test_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def validate_performance(
    model_path: str,
    test_data_path: str,
    thresholds: Dict[str, float],
    output_path: Optional[str] = None
) -> Tuple[bool, Dict]:
    """
    Validate model performance against thresholds.

    Args:
        model_path: Path to saved model
        test_data_path: Path to test data
        thresholds: Dictionary of metric thresholds
        output_path: Optional path to save results

    Returns:
        Tuple of (passed, metrics)
    """
    try:
        # Load model and data
        model = load_model(model_path)
        X_test, y_test = load_test_data(test_data_path)

        # Get predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_pred_proba)
        }

        # Check thresholds
        passed = all(
            metrics[metric] >= threshold
            for metric, threshold in thresholds.items()
        )

        # Save results if path provided
        if output_path:
            results = {
                "passed": passed,
                "metrics": metrics,
                "thresholds": thresholds
            }
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

        return passed, metrics

    except Exception as e:
        logger.error(f"Error validating model performance: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(
        description="Validate ML model performance."
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to saved model"
    )
    parser.add_argument(
        "--test-data-path",
        required=True,
        help="Path to test data"
    )
    parser.add_argument(
        "--threshold-accuracy",
        type=float,
        default=0.75,
        help="Minimum accuracy threshold"
    )
    parser.add_argument(
        "--threshold-auc",
        type=float,
        default=0.80,
        help="Minimum AUC threshold"
    )
    parser.add_argument(
        "--threshold-f1",
        type=float,
        default=0.70,
        help="Minimum F1 threshold"
    )
    parser.add_argument(
        "--output-path",
        help="Path to save results JSON"
    )

    args = parser.parse_args()

    # Set up thresholds
    thresholds = {
        "accuracy": args.threshold_accuracy,
        "auc": args.threshold_auc,
        "f1": args.threshold_f1
    }

    try:
        passed, metrics = validate_performance(
            args.model_path,
            args.test_data_path,
            thresholds,
            args.output_path
        )

        # Log results
        logger.info("Performance Validation Results:")
        for metric, value in metrics.items():
            threshold = thresholds.get(metric, "N/A")
            logger.info(f"{metric}: {value:.3f} (threshold: {threshold})")
        logger.info(f"Overall validation {'passed' if passed else 'failed'}")

        # Exit with status
        sys.exit(0 if passed else 1)

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
