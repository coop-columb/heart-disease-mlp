"""
Feature engineering for Heart Disease Prediction project.
"""

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_feature_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create feature interactions that might be relevant for heart disease prediction.

    Args:
        df: DataFrame containing the dataset

    Returns:
        DataFrame with added feature interactions
    """
    logger.info("Creating feature interactions")
    df = df.copy()

    # Age-gender interaction (age might have different effects based on gender)
    df["age_sex"] = df["age"] * df["sex"]

    # Chest pain and exercise angina combined risk
    if "cp" in df.columns and "exang" in df.columns:
        df["cp_exang"] = df["cp"] * df["exang"]

    # Blood pressure and cholesterol product (cardiovascular risk indicator)
    if "trestbps" in df.columns and "chol" in df.columns:
        df["bp_chol"] = df["trestbps"] * df["chol"] / 10000  # Scaled down

    # Heart rate reserve (if we have resting and max heart rate)
    if "thalach" in df.columns and "trestbps" in df.columns:
        df["heart_reserve"] = df["thalach"] - df["trestbps"]

    logger.info(
        f"Created new feature interactions: {list(set(df.columns) - set(list(df.columns)))}"
    )

    return df


def create_medical_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a medical risk score based on clinical knowledge.

    Args:
        df: DataFrame containing the dataset

    Returns:
        DataFrame with added risk score
    """
    logger.info("Creating medical risk score")
    df = df.copy()

    # Initialize risk score
    df["risk_score"] = 0

    # Age risk factor
    if "age" in df.columns:
        df.loc[df["age"] > 55, "risk_score"] += 1

    # Gender risk factor
    if "sex" in df.columns:
        df.loc[df["sex"] == 1, "risk_score"] += 1  # Male

    # Hypertension risk factor
    if "trestbps" in df.columns:
        df.loc[df["trestbps"] > 140, "risk_score"] += 1

    # Cholesterol risk factor
    if "chol" in df.columns:
        df.loc[df["chol"] > 240, "risk_score"] += 1

    # Blood sugar risk factor
    if "fbs" in df.columns:
        df.loc[df["fbs"] == 1, "risk_score"] += 1

    # Heart rate risk factor
    if "thalach" in df.columns:
        df.loc[df["thalach"] < 150, "risk_score"] += 1

    # Exercise-induced angina
    if "exang" in df.columns:
        df.loc[df["exang"] == 1, "risk_score"] += 1

    logger.info("Medical risk score created successfully")

    return df


def calculate_cardiovascular_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate cardiovascular age based on risk factors.

    Args:
        df: DataFrame containing the dataset

    Returns:
        DataFrame with added cardiovascular age feature
    """
    logger.info("Calculating cardiovascular age")
    df = df.copy()

    # Base cardiovascular age is chronological age
    df["cardiovascular_age"] = df["age"].copy()

    # Add years based on risk factors

    # High blood pressure
    if "trestbps" in df.columns:
        df.loc[df["trestbps"] > 140, "cardiovascular_age"] += 3
        df.loc[df["trestbps"] > 160, "cardiovascular_age"] += 5

    # High cholesterol
    if "chol" in df.columns:
        df.loc[df["chol"] > 240, "cardiovascular_age"] += 2
        df.loc[df["chol"] > 280, "cardiovascular_age"] += 4

    # Diabetes/high blood sugar
    if "fbs" in df.columns:
        df.loc[df["fbs"] == 1, "cardiovascular_age"] += 3

    # Exercise-induced angina
    if "exang" in df.columns:
        df.loc[df["exang"] == 1, "cardiovascular_age"] += 4

    logger.info("Cardiovascular age calculated successfully")

    return df


def normalize_categorical_levels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize categorical feature levels to ensure consistency.

    Args:
        df: DataFrame containing the dataset

    Returns:
        DataFrame with standardized categorical levels
    """
    logger.info("Normalizing categorical feature levels")
    df = df.copy()

    # Ensure categorical features are properly typed
    if "cp" in df.columns:
        df["cp"] = df["cp"].astype(int)

    if "restecg" in df.columns:
        df["restecg"] = df["restecg"].astype(int)

    if "slope" in df.columns:
        df["slope"] = df["slope"].astype(int)

    if "ca" in df.columns:
        # Handle string values that might be present
        df["ca"] = pd.to_numeric(df["ca"], errors="coerce")

    if "thal" in df.columns:
        # Handle string values that might be present
        df["thal"] = pd.to_numeric(df["thal"], errors="coerce")

    logger.info("Categorical feature levels normalized")

    return df


def select_features(
    df: pd.DataFrame, target_col: str = "target", additional_features: List[str] = None
) -> pd.DataFrame:
    """
    Select relevant features for heart disease prediction.

    Args:
        df: DataFrame containing the dataset
        target_col: Name of the target column
        additional_features: List of additional engineered features to keep

    Returns:
        DataFrame with selected features
    """
    logger.info("Selecting features for modeling")

    # Core features that are commonly used for heart disease prediction
    core_features = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
    ]

    # Add engineered features if specified
    if additional_features is not None:
        features = core_features + additional_features
    else:
        features = core_features

    # Add target column
    features.append(target_col)

    # Select features from DataFrame
    selected_df = df[features].copy()

    logger.info(f"Selected {len(features)} features for modeling")

    return selected_df
