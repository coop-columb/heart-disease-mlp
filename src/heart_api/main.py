import logging
import os
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.models.predict_model import HeartDiseasePredictor
from src.utils import load_config

from .api.endpoints import router as api_router
from .auth import AuthHandler, AuthSettings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("api.log")],
)
logger = logging.getLogger(__name__)

# Load configuration
config = load_config()

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

# Create FastAPI app
app = FastAPI(
    title="Heart Disease Prediction API",
    description="API for predicting heart disease risk based on clinical parameters",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Check if static directory exists, create if not
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Include API router
app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn

    # Get API configuration from config
    host = config["api"]["host"]
    port = config["api"]["port"]

    logger.info(f"Starting Heart Disease Prediction API on {host}:{port}")
    uvicorn.run("heart_api.main:app", host=host, port=port, reload=True)
