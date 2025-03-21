#!/bin/bash
# Download and version the dataset

# Verify current directory
if [ ! -d "src" ] || [ ! -d "data" ]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Create data directories if they don't exist
mkdir -p data/raw data/processed data/external

# Run data acquisition script
python src/data/download_data.py

echo "Data download complete. Raw data is in data/raw/"
