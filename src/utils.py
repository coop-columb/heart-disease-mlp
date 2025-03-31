"""
Utility functions for the Heart Disease Prediction project.
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_environment() -> str:
    """
    Get the current environment from the ENVIRONMENT environment variable.

    Returns:
        String representing the current environment: 'dev', 'staging', or 'prod'
    """
    # Get environment from environment variable, default to 'dev'
    env = os.environ.get("ENVIRONMENT", "dev").lower()

    # Validate environment
    if env not in ["dev", "development", "staging", "stage", "prod", "production"]:
        logger.warning(f"Unknown environment '{env}', falling back to 'dev'")
        env = "dev"

    # Normalize environment name
    if env in ["development", "dev"]:
        return "dev"
    elif env in ["stage", "staging"]:
        return "staging"
    elif env in ["prod", "production"]:
        return "prod"

    return env


def resolve_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve environment variables in configuration values.

    This function recursively traverses the configuration dictionary and
    replaces ${ENV_VAR} patterns with their corresponding environmen
    variable values.

    Args:
        config: Configuration dictionary

    Returns:
        Configuration with environment variables resolved
    """
    if not isinstance(config, dict):
        return config

    result = {}
    for key, value in config.items():
        if isinstance(value, dict):
            # Recursively process nested dictionaries
            result[key] = resolve_env_vars(value)
        elif isinstance(value, list):
            # Process each item in the list
            result[key] = [
                resolve_env_vars(item) if isinstance(item, dict) else item for item in value
            ]
        elif isinstance(value, str):
            # Replace environment variables in strings
            pattern = r"\${([A-Za-z0-9_]+)}"
            matches = re.findall(pattern, value)

            if matches:
                result_value = value
                for match in matches:
                    env_var = os.environ.get(match)
                    if env_var is not None:
                        result_value = result_value.replace(f"${{{match}}}", env_var)
                    else:
                        logger.warning(f"Environment variable '{match}' not found")
                result[key] = result_value
            else:
                result[key] = value
        else:
            # Keep other values as-is
            result[key] = value

    return result


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file based on the current environment.

    Args:
        config_path: Optional path to a specific configuration file
                     If not provided, environment-specific config is used

    Returns:
        Dictionary containing configuration parameters
    """
    if config_path is None:
        # Determine environment
        env = get_environment()

        # Load environment-specific configuration
        config_dir = Path("config")
        config_file = f"config.{env}.yaml"
        config_path = str(config_dir / config_file)

        # Fall back to default config if environment-specific config doesn't exist
        if not Path(config_path).exists():
            logger.warning(
                f"Configuration file '{config_path}' not found, falling back to config.yaml"
            )
            config_path = str(config_dir / "config.yaml")

    # Load configuration file
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Resolve environment variables in configuration
    config = resolve_env_vars(config)

    return config
