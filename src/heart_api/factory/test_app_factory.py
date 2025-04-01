import logging

import yaml
from fastapi import FastAPI

from heart_api.api.endpoints import router
from heart_api.auth import AuthHandler, AuthSettings
from heart_api.core import model_predictor

logger = logging.getLogger(__name__)


def load_test_config():
    with open("config/test_config.yaml", "r") as f:
        return yaml.safe_load(f)


def create_test_app() -> FastAPI:
    config = load_test_config()

    app = FastAPI()

    # Initialize state if it doesn't exist
    if not hasattr(app, "state"):
        app.state = type("State", (), {})()

    # Use model predictor from core

    # Add model predictor to app state
    logger.info("Setting up model predictor in app state")
    app.state.model_predictor = model_predictor

    if not hasattr(app.state, "model_predictor"):
        logger.error("Failed to set model_predictor in app state")
    else:
        logger.info(
            f"Successfully set model_predictor in app state. "
            f"Models loaded: sklearn={model_predictor.has_sklearn_model}, "
            f"keras={model_predictor.has_keras_model}"
        )

    # Initialize auth settings and handler
    auth_settings = AuthSettings(config)
    app.auth_handler = AuthHandler(settings=auth_settings)

    app.include_router(router)
    return app
