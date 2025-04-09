#!/usr/bin/env python3
"""
Validate ML data quality and characteristics.

This script:
1. Checks for missing values
2. Validates data distributions
3. Checks feature correlations
4. Ensures data quality standards
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def check_missing_values(
    df: pd.DataFrame,
    threshold: float = 0.05
) -> Tuple[bool, Dict]:
    """
    Check for missing values in dataset.

    Args:
        df: Input DataFrame
        threshold: Maximum allowed missing value percentage

    Returns:
        Tuple of (passed, details)
    """
    missing_counts = df.isnull().sum()
    missing_percentages = missing_counts / len(df)
    
    details = {
        col: {
            "missing_count": int(count),
            "missing_percentage": float(pct)
        }
        for col, count, pct in zip(
            df.columns,
            missing_counts,
            missing_percentages
        )
    }
    
    passed = all(pct <= threshold for pct in missing_percentages)
    
    return passed, details

def check_distributions(
    df: pd.DataFrame,
    reference_stats: Optional[Dict] = None
) -> Tuple[bool, Dict]:
    """
    Check data distributions against reference or for anomalies.

    Args:
        df: Input DataFrame
        reference_stats: Optional reference statistics

    Returns:
        Tuple of (passed, details)
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    distribution_checks = {}
    passed = True
    
    for col in numeric_cols:
        # Calculate basic statistics
        stats_dict = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "skew": float(stats.skew(df[col].dropna())),
            "kurtosis": float(stats.kurtosis(df[col].dropna())),
            "min": float(df[col].min()),
            "max": float(df[col].max())
        }
        
        # Check against reference if provided
        if reference_stats and col in reference_stats:
            ref = reference_stats[col]
            # Allow 10% deviation from reference
            stats_passed = all(
                abs(stats_dict[k] - ref[k]) <= 0.1 * abs(ref[k])
                for k in ["mean", "std"]
            )
            stats_dict["matches_reference"] = stats_passed
            passed &= stats_passed
        
        distribution_checks[col] = stats_dict
    
    return passed, distribution_checks

def check_correlations(
    df: pd.DataFrame,
    threshold: float = 0.9
) -> Tuple[bool, Dict]:
    """
    Check for high correlations between features.

    Args:
        df: Input DataFrame
        threshold: Maximum allowed correlation

    Returns:
        Tuple of (passed, details)
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    high_correlations = []
    passed = True
    
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            correlation = abs(corr_matrix.iloc[i, j])
            if correlation > threshold:
                passed = False
                high_correlations.append({
                    "feature1": numeric_cols[i],
                    "feature2": numeric_cols[j],
                    "correlation": float(correlation)
                })
    
    return passed, {
        "high_correlations": high_correlations,
        "max_correlation": float(
            max(abs(corr_matrix.max())) if len(numeric_cols) > 1 else 0
        )
    }

def validate_data(
    data_path: str,
    missing_threshold: float = 0.05,
    correlation_threshold: float = 0.9,
    reference_stats_path: Optional[str] = None,
    output_path: Optional[str] = None
) -> Tuple[bool, Dict]:
    """
    Run all data validation checks.

    Args:
        data_path: Path to data file
        missing_threshold: Maximum allowed missing values
        correlation_threshold: Maximum allowed correlation
        reference_stats_path: Optional path to reference statistics
        output_path: Optional path to save results

    Returns:
        Tuple of (passed, results)
    """
    try:
        # Load data
        df = pd.read_csv(data_path)
        
        # Load reference stats if provided
        reference_stats = None
        if reference_stats_path:
            with open(reference_stats_path) as f:
                reference_stats = json.load(f)
        
        # Run checks
        missing_passed, missing_details = check_missing_values(
            df, missing_threshold
        )
        dist_passed, dist_details = check_distributions(
            df, reference_stats
        )
        corr_passed, corr_details = check_correlations(
            df, correlation_threshold
        )
        
        # Combine results
        results = {
            "missing_values": {
                "passed": missing_passed,
                "details": missing_details
            },
            "distributions": {
                "passed": dist_passed,
                "details": dist_details
            },
            "correlations": {
                "passed": corr_passed,
                "details": corr_details
            }
        }
        
        # Overall pass/fail
        passed = all([
            missing_passed,
            dist_passed,
            corr_passed
        ])
        
        # Save results if path provided
        if output_path:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
        
        return passed, results
    
    except Exception as e:
        logger.error(f"Error validating data: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(
        description="Validate ML data quality."
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to data file"
    )
    parser.add_argument(
        "--missing-threshold",
        type=float,
        default=0.05,
        help="Maximum allowed missing value percentage"
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.9,
        help="Maximum allowed correlation"
    )
    parser.add_argument(
        "--reference-stats",
        help="Path to reference statistics JSON"
    )
    parser.add_argument(
        "--output-path",
        help="Path to save results JSON"
    )
    
    args = parser.parse_args()
    
    try:
        passed, results = validate_data(
            args.data_path,
            args.missing_threshold,
            args.correlation_threshold,
            args.reference_stats,
            args.output_path
        )
        
        # Log results
        logger.info("Data Validation Results:")
        logger.info(f"Missing Values Check: {'passed' if results['missing_values']['passed'] else 'failed'}")
        logger.info(f"Distribution Check: {'passed' if results['distributions']['passed'] else 'failed'}")
        logger.info(f"Correlation Check: {'passed' if results['correlations']['passed'] else 'failed'}")
        logger.info(f"Overall validation {'passed' if passed else 'failed'}")
        
        # Exit with status
        sys.exit(0 if passed else 1)
    
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
