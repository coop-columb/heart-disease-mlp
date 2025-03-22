#!/bin/bash
# Script to simulate a Render.com deployment locally
# This helps debug any issues before actual deployment

set -e

echo "==============================================" 
echo "Simulating Render.com deployment environment"
echo "=============================================="

# Build the Docker image (similar to what Render would do)
echo "Building Docker image..."
docker build -t heart-disease-render-test .

# Run container with the same environment variables as Render
echo "Running container with Render environment..."
docker run -it --rm \
  -p 8000:8000 \
  -e PORT=8000 \
  -e MODEL_DIR=/app/models \
  heart-disease-render-test

# Note: the container will stay running until you press Ctrl+C
# You can access the API at http://localhost:8000/health
# Check the logs for any errors that would occur in Render