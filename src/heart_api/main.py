import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from heart_api.api.endpoints import router as api_router
from heart_api.core import config, logger
from models.predict_model import HeartDiseasePredictor

# Initialize the model predictor
model_predictor = HeartDiseasePredictor()

# Create FastAPI app
app = FastAPI(
    title="Heart Disease Prediction API",
    description="API for predicting heart disease risk based on clinical parameters",
    version="1.0.0",
)


@app.get("/")
def root():
    """Root endpoint."""
    return """<!DOCTYPE html>
<html>
    <head><title>Heart Disease Prediction</title></head>
    <body>
        <h1>Heart Disease Prediction</h1>
        <p>API for predicting heart disease risk based on clinical parameters</p>
    </body>
</html>"""


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
