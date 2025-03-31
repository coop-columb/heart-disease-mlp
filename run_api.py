#!/usr/bin/env python
"""
Robust API launcher script for the Heart Disease Prediction API.
This ensures proper Python path isolation and handling of dependencies.
"""

import logging
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Clean Python path to avoid conflicts with other projects
logger.info("Setting up isolated Python environment...")
original_path = sys.path.copy()

# Filter out any paths related to other projects that might cause conflicts
filtered_path = [p for p in sys.path if "EmotionAdaptiveMusic" not in p]
sys.path = filtered_path

# Add the project root first to ensure it takes precedence
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root in sys.path:
    sys.path.remove(project_root)
sys.path.insert(0, project_root)

logger.info(f"Project root added to Python path: {project_root}")
logger.info("Python path cleaned to avoid conflicts")

# Now we can safely import our modules
import uvicorn  # noqa: E402

# Import app for ASGI reference
from api.app import app  # noqa: F401, E402

# Import utility functions for environment-specific configuration
from src.utils import get_environment, load_config  # noqa: E402

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Heart Disease Prediction API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the API server")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the API server")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--env", help="Environment to use (dev, staging, prod)")

    args = parser.parse_args()

    # Set environment from command line argument if provided
    if args.env:
        os.environ["ENVIRONMENT"] = args.env

    # Get current environmen
    env = get_environment()

    # Load environment-specific configuration
    config = load_config()

    # Override reload setting from environment config if not explicitly se
    if not args.reload and "environment" in config and "reload" in config["environment"]:
        reload_setting = config["environment"]["reload"]
    else:
        reload_setting = args.reload

    # Set log level based on configuration
    log_level = config.get("api", {}).get("log_level", "info").upper()
    logging.basicConfig(level=getattr(logging, log_level))

    logger.info(f"Starting Heart Disease Prediction API in {env.upper()} environment")
    logger.info(f"Host: {args.host}, Port: {args.port}, Reload: {reload_setting}")
    logger.info(f"Using configuration: config/config.{env}.yaml")
    logger.info("API Documentation available at http://localhost:8000/docs")

    uvicorn.run(
        "api.app:app",
        host=args.host,
        port=args.port,
        reload=reload_setting,
        log_level=log_level.lower(),
    )
