"""
Utility functions for the Heart Disease Prediction project.
"""

from typing import Any, Dict

import yaml


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Dictionary containing configuration parameters
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config
