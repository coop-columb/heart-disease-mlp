#\!/bin/bash
# Deploy heart disease prediction API

# Verify current directory
if [ \! -d "src" ] || [ \! -d "api" ]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

# Check if Docker is installed
if \! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if \! command -v docker-compose &> /dev/null; then
    echo "Error: Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if required models exist
if [ \! -f "models/sklearn_mlp_model.joblib" ] || [ \! -f "models/keras_mlp_model.h5" ]; then
    echo "Models not found. Training models first..."
    ./scripts/train_models.sh
fi

# Build and start the Docker container
echo "Building and starting Docker container..."
docker-compose up -d --build

echo "API deployment completed successfully."
echo "API is running at http://localhost:8000"
echo "Documentation available at http://localhost:8000/docs"
