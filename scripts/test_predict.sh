#!/bin/bash

# Configuration
API_URL="http://localhost:8000"
API_KEY="dev_api_key"  # Development API key

# Get authentication token
echo "Getting authentication token..."
TOKEN_RESPONSE=$(curl -s -X POST "${API_URL}/auth/token" \
    -H "X-API-Key: ${API_KEY}")

# Check if token request was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to connect to the server"
    exit 1
fi

# Extract token
TOKEN=$(echo ${TOKEN_RESPONSE} | grep -o '"access_token":"[^"]*"' | cut -d'"' -f4)

if [ -z "$TOKEN" ]; then
    echo "Error: Failed to get token. Response: ${TOKEN_RESPONSE}"
    exit 1
fi

echo "Successfully obtained token."

# Make prediction request
echo "Sending prediction request..."
PREDICTION_RESPONSE=$(curl -s -X POST "${API_URL}/predict" \
    -H "Authorization: Bearer ${TOKEN}" \
    -H "Content-Type: application/json" \
    -d '{
    "age": 61,
    "sex": 1,
    "cp": 3,
    "trestbps": 140,
    "chol": 240,
    "fbs": 1,
    "restecg": 1,
    "thalach": 150,
    "exang": 1,
    "oldpeak": 2.4,
    "slope": 2,
    "ca": 1,
    "thal": 3
}')

if [ $? -ne 0 ]; then
    echo "Error: Failed to make prediction request"
    exit 1
fi

echo "Prediction Results:"
echo "${PREDICTION_RESPONSE}"
