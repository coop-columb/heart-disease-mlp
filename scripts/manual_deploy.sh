#!/bin/bash
# Simple manual deployment script for Heart Disease Prediction API
# This script builds and pushes the Docker image to a registry
# You can then deploy the image to any platform that supports Docker

set -e

# Configuration
IMAGE_NAME="heart-disease-mlp"
TAG=$(git rev-parse --short HEAD)
FULL_IMAGE_NAME="$IMAGE_NAME:$TAG"

echo "Building Docker image: $FULL_IMAGE_NAME"
docker build -t $FULL_IMAGE_NAME .

echo "Running basic tests..."
docker run --rm $FULL_IMAGE_NAME pytest -xvs tests/test_models.py::test_build_sklearn_mlp

echo "You can run the image locally with:"
echo "docker run -p 8000:8000 $FULL_IMAGE_NAME"

echo ""
echo "To push to a Docker registry (after docker login):"
echo "docker tag $FULL_IMAGE_NAME your-registry/heart-disease-mlp:latest"
echo "docker push your-registry/heart-disease-mlp:latest"

echo ""
echo "For Render.com deployment:"
echo "1. Go to: https://dashboard.render.com/new/web-service"
echo "2. Choose 'Deploy an existing image from a registry'"
echo "3. Enter image: your-registry/heart-disease-mlp:latest"
echo "4. Set environment variables:"
echo "   - PORT=8000"
echo "   - MODEL_DIR=/app/models"

echo ""
echo "Deployment documentation at: .github/SIMPLIFIED_DEPLOYMENT.md"