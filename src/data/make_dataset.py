"""
Script to prepare the heart disease dataset for modeling.
"""

import argparse
import logging
import os

import joblib
import numpy as np
import pandas as pd

from src.data.preprocess import (
    binarize_target,
    create_preprocessing_pipeline,
    handle_missing_values,
    load_data,
    split_data,
)
from src.features.feature_engineering import (
    calculate_cardiovascular_age,
    create_feature_interactions,
    create_medical_risk_score,
    normalize_categorical_levels,
    select_features,
)
from src.utils import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main(config_path, raw_data_path=None, processed_dir=None):
    """
    Run data processing scripts to turn raw data into processed data for modeling.

    Args:
        config_path: Path to configuration file
        raw_data_path: Path to raw data (overrides config if provided)
        processed_dir: Directory to save processed data (overrides config if provided)
    """
    # Load configuration
    config = load_config(config_path)

    # Override paths if provided
    if raw_data_path is None:
        raw_data_path = config["data"]["raw_data_path"]

    if processed_dir is None:
        processed_dir = os.path.dirname(config["data"]["processed_data_path"])

    logger.info(f"Processing data from {raw_data_path}")

    # Create processed directory if it doesn't exist
    os.makedirs(processed_dir, exist_ok=True)

    # Load data
    df = load_data(raw_data_path)

    # Extract configuration parameters
    categorical_features = config["preprocessing"]["categorical_features"]
    numerical_features = config["preprocessing"]["numerical_features"]
    target = config["preprocessing"]["target"]
    test_size = config["preprocessing"]["test_size"]
    validation_size = config["preprocessing"]["validation_size"]
    random_seed = config["preprocessing"]["random_seed"]

    # Preprocess data
    df = binarize_target(df, target_col=target)
    df = handle_missing_values(df)
    df = normalize_categorical_levels(df)

    # Feature engineering
    df = create_feature_interactions(df)
    df = create_medical_risk_score(df)
    df = calculate_cardiovascular_age(df)

    # Select features for modeling
    additional_features = [
        "age_sex",
        "cp_exang",
        "bp_chol",
        "risk_score",
        "cardiovascular_age",
    ]
    df = select_features(df, target_col=target, additional_features=additional_features)

    # Update numerical features with new engineered features
    engineered_numerical = [
        col
        for col in df.columns
        if col not in categorical_features + [target]
        and col not in numerical_features
        and df[col].dtype in ["int64", "float64"]
    ]

    all_numerical_features = numerical_features + engineered_numerical
    logger.info(f"Added {len(engineered_numerical)} engineered numerical features")

    # Split features and target
    X = df.drop(target, axis=1)
    y = df[target]

    # Split data into train/val/test sets
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, test_size=test_size, val_size=validation_size, random_state=random_seed
    )

    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(
        categorical_features=categorical_features,
        numerical_features=all_numerical_features,
        numeric_imputer="median",
        use_robust_scaler=True,
    )

    # Fit preprocessor on training data and transform all sets
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    # Save processed data
    processed_data_path = os.path.join(processed_dir, "processed_data.npz")
    np.savez(
        processed_data_path,
        X_train=X_train_processed,
        X_val=X_val_processed,
        X_test=X_test_processed,
        y_train=y_train.values,
        y_val=y_val.values,
        y_test=y_test.values,
    )
    logger.info(f"Saved processed data to {processed_data_path}")

    # Save preprocessor
    preprocessor_path = os.path.join(processed_dir, "preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_path)
    logger.info(f"Saved preprocessor to {preprocessor_path}")

    # Save original data splits for reference
    original_data_path = os.path.join(processed_dir, "original_splits.joblib")
    joblib.dump(
        {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
        },
        original_data_path,
    )
    logger.info(f"Saved original data splits to {original_data_path}")

    # Create metadata file
    metadata = {
        "original_data_path": raw_data_path,
        "train_samples": X_train.shape[0],
        "validation_samples": X_val.shape[0],
        "test_samples": X_test.shape[0],
        "total_samples": X_train.shape[0] + X_val.shape[0] + X_test.shape[0],
        "numerical_features": all_numerical_features,
        "categorical_features": categorical_features,
        "engineered_features": engineered_numerical,
        "feature_count_before_encoding": X_train.shape[1],
        "feature_count_after_encoding": X_train_processed.shape[1],
        "positive_class_ratio_train": y_train.mean(),
        "positive_class_ratio_val": y_val.mean(),
        "positive_class_ratio_test": y_test.mean(),
    }

    metadata_path = os.path.join(processed_dir, "processing_metadata.txt")
    with open(metadata_path, "w") as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    logger.info(f"Saved processing metadata to {metadata_path}")

    logger.info("Dataset preparation complete.")

    return preprocessor, X_train_processed.shape[1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare heart disease dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--raw-data-path",
        type=str,
        default=None,
        help="Path to raw data (overrides config if provided)",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=None,
        help="Directory to save processed data (overrides config if provided)",
    )

    args = parser.parse_args()
    main(args.config, args.raw_data_path, args.processed_dir)
