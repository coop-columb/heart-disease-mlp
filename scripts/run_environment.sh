#!/bin/bash
# Script to run the Heart Disease Prediction system in different environments

set -e

# Determine the project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Default environment
ENV="dev"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --env=*)
            ENV="${1#*=}"
            shift
            ;;
        --help)
            echo "Usage: $0 [--env=ENVIRONMENT]"
            echo
            echo "Options:"
            echo "  --env=ENV  Set the environment: dev, staging, prod (default: dev)"
            echo "  --help     Show this help message"
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
    fi
fi

# Generate .env file for the selected environment if it doesn't exist
if [ ! -f ".env.${ENV}" ]; then
    echo "Generating .env file for ${ENV} environment..."
    python scripts/generate_env_file.py "${ENV}"
fi

# Export environment variables from .env file
echo "Loading environment variables from .env.${ENV}..."
export $(grep -v '^#' .env.${ENV} | xargs)

# Set ENVIRONMENT variable for use in scripts
export ENVIRONMENT="${ENV}"

# Start the system using Docker Compose
echo "Starting the Heart Disease Prediction system in ${ENV} environment..."
docker-compose -f docker-compose.${ENV}.yaml up -d

echo
echo "System started in ${ENV} environment!"
echo "API is available at http://localhost:8000"
echo "Documentation: http://localhost:8000/docs"
echo
echo "To view logs: docker-compose -f docker-compose.${ENV}.yaml logs -f"
echo "To stop the system: docker-compose -f docker-compose.${ENV}.yaml down"