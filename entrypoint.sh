#!/bin/bash
set -e

# Create necessary directories
mkdir -p data/processed models

# Start the API server
exec uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8000}
