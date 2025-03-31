#!/bin/bash
set -e

# Print environment for debugging
echo "Starting entrypoint script..."
echo "PORT: ${PORT}"
echo "MODEL_DIR: ${MODEL_DIR}"
echo "ENVIRONMENT: ${ENVIRONMENT}"

# Create necessary directories
mkdir -p data/processed models backups

# Print directory structure for debugging
echo "Directory structure:"
ls -la

# Print available configuration files
echo "Available configuration files:"
ls -la config/

# Use the run_api.py script to handle environment-specific configuration
echo "Starting API server on port ${PORT:-8000} in ${ENVIRONMENT:-dev} environment..."

# Default to first argument if provided, otherwise use python run_api.py
if [ $# -eq 0 ]; then
    exec python run_api.py --host 0.0.0.0 --port ${PORT:-8000}
else
    exec "$@"
fi
