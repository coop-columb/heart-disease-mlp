#\!/bin/bash
# Run API tests

# Verify current directory
if [ \! -d "src" ] || [ \! -d "api" ]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Run API tests
echo "Running API tests..."
pytest tests/test_api.py -v

# Display results
if [ $? -eq 0 ]; then
    echo "All API tests passed successfully\!"
else
    echo "Some API tests failed. Please check the output above for details."
fi
