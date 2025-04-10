#!/usr/bin/env python3
"""
Check for data leakage in ML pipeline.

This script:
1. Validates train/test data separation
2. Checks for feature leakage
3. Validates temporal data splits
4. Verifies preprocessing is applied correctly
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def check_data_separation(
    train_data: pd.DataFrame, 
    test_data: pd.DataFrame,
    id_columns: List[str] = None
) -> Tuple[bool, Dict]:
    """
    Check if train and test data are properly separated.
    
    Args:
        train_data: Training dataset
        test_data: Testing dataset
        id_columns: List of ID columns to check for overlap
        
    Returns:
        Tuple of (passed, details)
    """
    details = {
        "duplicate_rows": 0,
        "id_overlaps": {},
        "issues": []
    }
    
    # Check for duplicate rows
    if train_data.shape[1] != test_data.shape[1]:
        details["issues"].append("Train and test data have different number of columns")
    else:
        # Convert to string for comparison to handle different data types
        train_set = set(train_data.astype(str).apply(tuple, axis=1))
        test_set = set(test_data.astype(str).apply(tuple, axis=1))
        overlaps = train_set.intersection(test_set)
        
        details["duplicate_rows"] = len(overlaps)
        if len(overlaps) > 0:
            details["issues"].append(f"Found {len(overlaps)} duplicate rows between train and test data")
    
    # Check ID column overlaps if provided
    if id_columns:
        for col in id_columns:
            if col in train_data.columns and col in test_data.columns:
                train_ids = set(train_data[col])
                test_ids = set(test_data[col])
                overlap = train_ids.intersection(test_ids)
                
                details["id_overlaps"][col] = len(overlap)
                if len(overlap) > 0:
                    details["issues"].append(f"Found {len(overlap)} overlapping IDs in column '{col}'")
    
    # Determine if validation passed
    passed = len(details["issues"]) == 0
    
    return passed, details

def check_feature_leakage(
    data: pd.DataFrame,
    target_column: str,
    threshold: float = 0.8
) -> Tuple[bool, Dict]:
    """
    Check for features that leak information about the target.
    
    Args:
        data: Dataset with features and target
        target_column: Name of target column
        threshold: Threshold for correlation/mutual information
        
    Returns:
        Tuple of (passed, details)
    """
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")
    
    details = {
        "high_correlation_features": [],
        "high_mutual_info_features": [],
        "issues": []
    }
    
    # Get target values
    y = data[target_column]
    
    # Check correlations (for numeric features)
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != target_column:
            corr = abs(data[col].corr(data[target_column]))
            if corr > threshold:
                details["high_correlation_features"].append({
                    "feature": col,
                    "correlation": float(corr)
                })
                details["issues"].append(f"Feature '{col}' has high correlation ({corr:.3f}) with target")
    
    # Check mutual information (for all features)
    for col in data.columns:
        if col != target_column:
            # Convert to categorical for mutual info calculation if needed
            x = data[col]
            if not pd.api.types.is_numeric_dtype(x):
                x = pd.factorize(x)[0]
            
            y_factorized = pd.factorize(y)[0] if not pd.api.types.is_numeric_dtype(y) else y
            
            try:
                mi = mutual_info_score(x, y_factorized)
                normalized_mi = mi / np.log(len(data)) if len(data) > 1 else 0
                
                if normalized_mi > threshold:
                    details["high_mutual_info_features"].append({
                        "feature": col,
                        "mutual_info": float(normalized_mi)
                    })
                    details["issues"].append(f"Feature '{col}' has high mutual information ({normalized_mi:.3f}) with target")
            except Exception as e:
                logger.warning(f"Could not calculate mutual information for '{col}': {e}")
    
    # Determine if validation passed
    passed = len(details["issues"]) == 0
    
    return passed, details

def check_temporal_leakage(
    data: pd.DataFrame,
    time_column: str,
    target_column: str
) -> Tuple[bool, Dict]:
    """
    Check for temporal leakage in time-series data.
    
    Args:
        data: Time series dataset
        time_column: Name of timestamp column
        target_column: Name of target column
        
    Returns:
        Tuple of (passed, details)
    """
    if time_column not in data.columns:
        raise ValueError(f"Time column '{time_column}' not found in data")
    
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")
    
    details = {
        "issues": []
    }
    
    # Convert time column to datetime if needed
    if not pd.api.types.is_datetime64_dtype(data[time_column]):
        try:
            data[time_column] = pd.to_datetime(data[time_column])
        except Exception as e:
            details["issues"].append(f"Could not convert time column to datetime: {e}")
            return False, details
    
    # Sort by time
    data = data.sort_values(by=time_column)
    
    # Check for patterns indicating future leak
    # 1. Sharp discontinuities in target by time
    data['prev_target'] = data[target_column].shift(1)
    data['next_target'] = data[target_column].shift(-1)
    
    # Check if next target value strongly predicts current target
    next_target_corr = abs(data[target_column].corr(data['next_target']))
    prev_target_corr = abs(data[target_column].corr(data['prev_target']))
    
    if next_target_corr > 0.9 and next_target_corr > prev_target_corr * 1.5:
        details["issues"].append(f"Suspiciously high correlation with future target values ({next_target_corr:.3f})")
    
    # 2. Check for sudden jumps in target that align with time breaks
    data['time_diff'] = data[time_column].diff().dt.total_seconds()
    data['target_diff'] = abs(data[target_column].diff())
    
    # Identify large time gaps
    time_gaps = data[data['time_diff'] > data['time_diff'].quantile(0.95)]
    
    # Check if target changes significantly at time gaps
    if not time_gaps.empty:
        avg_target_diff = data['target_diff'].mean()
        gap_target_diff = time_gaps['target_diff'].mean()
        
        if gap_target_diff > avg_target_diff * 2:
            details["issues"].append(f"Target changes significantly at time breaks (ratio: {gap_target_diff/avg_target_diff:.2f})")
    
    # Determine if validation passed
    passed = len(details["issues"]) == 0
    
    return passed, details

def check_preprocessing_leakage(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame
) -> Tuple[bool, Dict]:
    """
    Check if preprocessing was correctly applied without leakage.
    
    Args:
        train_data: Training dataset
        test_data: Testing dataset
        
    Returns:
        Tuple of (passed, details)
    """
    details = {
        "issues": []
    }
    
    # Check for statistical differences between normalized/scaled features
    numeric_cols = train_data.select_dtypes(include=[np.number]).columns.intersection(
        test_data.select_dtypes(include=[np.number]).columns
    )
    
    for col in numeric_cols:
        train_mean = train_data[col].mean()
        test_mean = test_data[col].mean()
        train_std = train_data[col].std()
        test_std = test_data[col].std()
        
        # If data is standardized, mean should be near 0 and std near 1 for both datasets
        if (-0.1 < train_mean < 0.1 and 0.9 < train_std < 1.1):
            # Check if test data has similar stats, indicating shared preprocessing
            if not (-0.1 < test_mean < 0.1 and 0.9 < test_std < 1.1):
                details["issues"].append(f"Column '{col}' appears to be standardized in train but not in test data")
        
        # Check for suspicious statistical differences
        if train_std > 0 and test_std > 0:
            mean_diff_normalized = abs(train_mean - test_mean) / ((train_std + test_std) / 2)
            std_ratio = max(train_std / test_std, test_std / train_std)
            
            if mean_diff_normalized > 0.5 and std_ratio > 1.5:
                details["issues"].append(f"Column '{col}' has suspicious statistical differences between train and test")
    
    # Determine if validation passed
    passed = len(details["issues"]) == 0
    
    return passed, details

def check_data_leakage(
    train_path: str,
    test_path: str,
    target_column: str,
    time_column: Optional[str] = None,
    id_columns: Optional[List[str]] = None,
    leakage_threshold: float = 0.8,
    output_path: Optional[str] = None
) -> Tuple[bool, Dict]:
    """
    Run comprehensive data leakage checks.
    
    Args:
        train_path: Path to training data
        test_path: Path to test data
        target_column: Name of target column
        time_column: Optional name of timestamp column
        id_columns: Optional list of ID columns
        leakage_threshold: Threshold for feature leakage detection
        output_path: Optional path to save results
        
    Returns:
        Tuple of (passed, results)
    """
    try:
        # Load data
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        
        # Run checks
        separation_passed, separation_details = check_data_separation(
            train_data, test_data, id_columns
        )
        
        feature_passed, feature_details = check_feature_leakage(
            train_data, target_column, leakage_threshold
        )
        
        preprocessing_passed, preprocessing_details = check_preprocessing_leakage(
            train_data, test_data
        )
        
        # Run temporal check if time column provided
        temporal_passed = True
        temporal_details = {"issues": ["Temporal check skipped (no time column provided)"]}
        
        if time_column:
            temporal_passed, temporal_details = check_temporal_leakage(
                pd.concat([train_data, test_data]), time_column, target_column
            )
        
        # Combine results
        results = {
            "data_separation": {
                "passed": separation_passed,
                "details": separation_details
            },
            "feature_leakage": {
                "passed": feature_passed,
                "details": feature_details
            },
            "temporal_leakage": {
                "passed": temporal_passed,
                "details": temporal_details
            },
            "preprocessing_leakage": {
                "passed": preprocessing_passed,
                "details": preprocessing_details
            }
        }
        
        # Overall pass/fail
        passed = all([
            separation_passed,
            feature_passed,
            temporal_passed,
            preprocessing_passed
        ])
        
        # Save results if path provided
        if output_path:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
        
        return passed, results
    
    except Exception as e:
        logger.error(f"Error checking data leakage: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(
        description="Check for data leakage in ML pipeline."
    )
    parser.add_argument(
        "--train-path",
        required=True,
        help="Path to training data CSV"
    )
    parser.add_argument(
        "--test-path",
        required=True,
        help="Path to test data CSV"
    )
    parser.add_argument(
        "--target-column",
        required=True,
        help="Name of target column"
    )
    parser.add_argument(
        "--time-column",
        help="Name of timestamp column for temporal checks"
    )
    parser.add_argument(
        "--id-columns",
        help="Comma-separated list of ID columns to check for overlap"
    )
    parser.add_argument(
        "--leakage-threshold",
        type=float,
        default=0.8,
        help="Threshold for feature leakage detection"
    )
    parser.add_argument(
        "--output-path",
        help="Path to save results JSON"
    )
    
    args = parser.parse_args()
    
    # Parse ID columns if provided
    id_columns = args.id_columns.split(",") if args.id_columns else None
    
    try:
        passed, results = check_data_leakage(
            args.train_path,
            args.test_path,
            args.target_column,
            args.time_column,
            id_columns,
            args.leakage_threshold,
            args.output_path
        )
        
        # Log results
        logger.info("Data Leakage Check Results:")
        
        for check_name, check_results in results.items():
            logger.info(f"{check_name}: {'passed' if check_results['passed'] else 'failed'}")
            if not check_results['passed']:
                for issue in check_results['details'].get('issues', []):
                    logger.warning(f"- {issue}")
        
        # Exit with status
        sys.exit(0 if passed else 1)
