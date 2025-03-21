#!/bin/bash
# Train heart disease prediction models

# Verify current directory
if [ ! -d "src" ] || [ ! -d "data" ]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Create necessary directories
mkdir -p models reports/figures

# Check if processed data exists
if [ ! -f "data/processed/processed_data.npz" ]; then
    echo "Processed data not found. Processing data first..."
    ./scripts/process_data.sh
fi

# Perform hyperparameter tuning and model training
if [ "$1" == "--tune" ]; then
    echo "Training models with hyperparameter tuning (this may take a while)..."
    python src/models/train_model.py --tune
else
    echo "Training models with default parameters..."
    python src/models/train_model.py
fi

echo "Model training complete. Model files are in models/ directory."
echo "Evaluation reports and visualizations are in reports/ directory."
