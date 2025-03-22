#!/bin/bash
set -e

# Print environment for debugging
echo "Starting entrypoint script..."
echo "PORT: ${PORT}"
echo "MODEL_DIR: ${MODEL_DIR}"

# Create necessary directories
mkdir -p data/processed models

# Print directory structure for debugging
echo "Directory structure:"
ls -la

# Start the API server
echo "Starting API server on port ${PORT:-8000}..."
exec uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8000}
