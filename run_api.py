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

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Heart Disease Prediction API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the API server")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the API server")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")

    args = parser.parse_args()

    logger.info(f"Starting Heart Disease Prediction API on {args.host}:{args.port}")
    logger.info("API Documentation available at http://localhost:8000/docs")

    uvicorn.run("api.app:app", host=args.host, port=args.port, reload=args.reload)
