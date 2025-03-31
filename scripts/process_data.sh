#!/bin/bash
# Process the raw data for modeling

# Verify current directory
if [ ! -d "src" ] || [ ! -d "data" ]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if raw data exists
if [ ! -f "data/raw/heart_disease_combined.csv" ]; then
    echo "Raw data not found. Downloading data first..."
    ./scripts/get_data.sh
fi

# Create processed directory if it doesn't exist
mkdir -p data/processed

# Run data processing script
python src/data/make_dataset.py

echo "Data processing complete. Processed data is in data/processed/"
