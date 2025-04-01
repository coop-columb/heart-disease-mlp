import logging
import os
from concurrent.futures import ThreadPoolExecutor

from models.predict_model import HeartDiseasePredictor

from ..auth import AuthHandler, AuthSettings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("api.log")],
)
logger = logging.getLogger(__name__)

# Default configuration with explicit auth settings
config = {
    "api": {
        "auth": {
            "enabled": True,
            "secret_key": os.getenv("API_SECRET_KEY", "dev_secret_key"),
            "access_token_expire_minutes": 30,
            "public_endpoints": [
                "/docs",
                "/openapi.json",
                "/health",
                "/",
                "/version",
                "/metrics",
                "/models/info",
                "/predict",
                "/predict/batch",
                "/cache/stats",
                "/cache/config",
                "/cache/clear",
            ],
            "api_keys": [
                {"key": os.getenv("DEV_API_KEY", "dev_api_key")},
                {"key": "test_api_key_1"},
            ],
        }
    }
}

# Initialize model predictor
model_predictor = HeartDiseasePredictor(model_dir="models")

# Batch processing configuration
BATCH_SIZE = config.get("api", {}).get("batch_size", 50)
MAX_WORKERS = config.get("api", {}).get("max_workers", 4)
PERFORMANCE_LOGGING = config.get("api", {}).get("performance_logging", True)

# Cache configuration
cache_config = config.get("api", {}).get("caching", {})
CACHE_ENABLED = cache_config.get("enabled", True)
CACHE_MAX_SIZE = cache_config.get("max_size", 1000)
CACHE_TTL = cache_config.get("ttl", 3600)  # 1 hour default

# Initialize thread pool for parallel processing
thread_pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# Initialize authentication
auth_settings = AuthSettings(config)
auth_handler = AuthHandler(auth_settings)
