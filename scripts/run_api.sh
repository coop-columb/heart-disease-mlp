#!/bin/bash
# Robust script to run the Heart Disease Prediction API

set -e

# Determine the project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to the project root
cd "${PROJECT_ROOT}"

# Check if Python virtual environment exists and activate it
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    # Use the appropriate activation command based on your platform
    if [ -f "venv/bin/activate" ]; then
        # Linux/macOS
        source venv/bin/activate
    elif [ -f "venv/Scripts/activate" ]; then
        # Windows
        source venv/Scripts/activate
    else
        echo "Virtual environment found but activation script could not be located."
        echo "Proceeding without virtual environment activation."
    fi
fi

# Ensure required packages are installed
if [ -f "requirements.txt" ]; then
    echo "Checking required packages..."
    pip install -q -r requirements.txt
fi

# Set default host, port, and environment
HOST="0.0.0.0"
PORT="8000"
ENV="dev"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --host=*)
            HOST="${1#*=}"
            shift
            ;;
        --port=*)
            PORT="${1#*=}"
            shift
            ;;
        --env=*)
            ENV="${1#*=}"
            shift
            ;;
        --help)
            echo "Usage: $0 [--host=HOSTNAME] [--port=PORT] [--env=ENVIRONMENT]"
            echo
            echo "Options:"
            echo "  --host=HOSTNAME  Set the host address to bind to (default: 0.0.0.0)"
            echo "  --port=PORT      Set the port to listen on (default: 8000)"
            echo "  --env=ENV        Set the environment: dev, staging, prod (default: dev)"
            echo "  --help           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# Validate environment
if [[ "$ENV" != "dev" && "$ENV" != "staging" && "$ENV" != "prod" ]]; then
    echo "Invalid environment: $ENV"
    echo "Valid environments are: dev, staging, prod"
    exit 1
fi

# Export environment variable
export ENVIRONMENT="${ENV}"

# Print environment information
echo "Starting Heart Disease Prediction API with the following settings:"
echo "  Host: ${HOST}"
echo "  Port: ${PORT}"
echo "  Environment: ${ENV}"
echo "  Config: config/config.${ENV}.yaml"
echo "  API Documentation: http://localhost:${PORT}/docs"

# Run the API using the run_api.py script
python run_api.py --host "${HOST}" --port "${PORT}"
