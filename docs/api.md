# Heart Disease Prediction API Documentation

This document provides detailed information about the Heart Disease Prediction REST API, including endpoints, request/response formats, and example usage.

## Table of Contents

- [Overview](#overview)
- [Base URL](#base-url)
- [Authentication](#authentication)
- [Endpoints](#endpoints)
  - [Health Check](#health-check)
  - [Model Information](#model-information)
  - [Batch Configuration](#batch-configuration)
  - [Cache Management](#cache-management)
  - [Prediction](#prediction)
- [Request and Response Formats](#request-and-response-formats)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [Example Usage](#example-usage)
  - [Using cURL](#using-curl)
  - [Using Python](#using-python)
  - [Using JavaScript](#using-javascript)
- [API Testing Script](#api-testing-script)

## Overview

The Heart Disease Prediction API provides a REST interface to the machine learning models trained for predicting heart disease risk. It allows applications to submit patient medical parameters and receive predictions along with risk assessments and interpretations.

The API is built using FastAPI, providing automatic OpenAPI documentation, request validation, and efficient performance.

## Base URL

- **Local Development**: `http://localhost:8000`
- **Docker Deployment**: `http://localhost:8000`
- **Production**: Depends on your deployment (e.g., `https://api.heartdisease-prediction.com`)

## Authentication

Currently, the API does not implement authentication. For production deployments, you should implement an authentication mechanism (e.g., API keys, OAuth2, or JWT tokens).

## Endpoints

The API provides the following endpoints:

### Health Check

- **URL**: `/health`
- **Method**: `GET`
- **Description**: Verifies that the API service is running and can access its dependencies (models, preprocessor).
- **Response**: Status information about the API service and its dependencies.

**Example Response**:
```json
{
  "status": "healthy",
  "api_version": "1.0.0",
  "models_available": {
    "sklearn_mlp": true,
    "keras_mlp": true
  },
  "preprocessor_available": true
}
```

### Model Information

- **URL**: `/models/info`
- **Method**: `GET`
- **Description**: Provides information about the available models and their configuration.
- **Response**: Details about the loaded models, including their architecture, training date, and performance metrics.

**Example Response**:
```json
{
  "models_available": {
    "sklearn_mlp": true,
    "keras_mlp": true
  },
  "ensemble_available": true,
  "preprocessor_available": true,
  "model_info": {
    "sklearn_mlp": {
      "type": "MLPClassifier",
      "hidden_layer_sizes": [128, 64, 32],
      "activation": "relu",
      "solver": "adam",
      "training_date": "2023-09-15"
    },
    "keras_mlp": {
      "type": "Sequential",
      "layers": 5,
      "activation": "relu",
      "output_activation": "sigmoid",
      "training_date": "2023-09-15"
    }
  },
  "performance_metrics": {
    "sklearn_mlp": {
      "accuracy": 0.853,
      "precision": 0.858,
      "recall": 0.841,
      "f1": 0.849,
      "roc_auc": 0.929
    },
    "keras_mlp": {
      "accuracy": 0.853,
      "precision": 0.845,
      "recall": 0.870,
      "f1": 0.857,
      "roc_auc": 0.926
    },
    "ensemble": {
      "accuracy": 0.864,
      "precision": 0.861,
      "recall": 0.870,
      "f1": 0.865,
      "roc_auc": 0.930
    }
  }
}
```

### Batch Configuration

- **URL**: `/batch/config`
- **Method**: `GET`
- **Description**: Gets the current batch processing configuration.
- **Response**: Configuration parameters for batch prediction processing.

**Example Response**:
```json
{
  "batch_size": 50,
  "max_workers": 4,
  "performance_logging": true
}
```

- **URL**: `/batch/config`
- **Method**: `POST`
- **Description**: Updates the batch processing configuration.
- **Request**: JSON object containing batch processing parameters to update.
- **Response**: Updated configuration parameters for batch prediction processing.

**Example Request**:
```json
{
  "batch_size": 100,
  "max_workers": 8,
  "performance_logging": true
}
```

**Example Response**:
```json
{
  "batch_size": 100,
  "max_workers": 8,
  "performance_logging": true,
  "cache_config": {
    "enabled": true,
    "max_size": 1000,
    "ttl_seconds": 3600
  }
}
```

### Cache Management

- **URL**: `/cache/stats`
- **Method**: `GET`
- **Description**: Gets current prediction cache statistics.
- **Response**: Detailed statistics about the prediction cache.

**Example Response**:
```json
{
  "enabled": true,
  "max_size": 1000,
  "ttl_seconds": 3600,
  "entries": 45,
  "hits": 127,
  "misses": 62,
  "hit_rate": 0.672,
  "evictions": 0,
  "created_at": "2025-03-31T14:30:00.123456"
}
```

- **URL**: `/cache/config`
- **Method**: `POST`
- **Description**: Updates the prediction cache configuration.
- **Request**: JSON object containing cache configuration parameters to update.
- **Response**: Updated configuration and statistics for the prediction cache.

**Example Request**:
```json
{
  "enabled": true,
  "max_size": 2000,
  "ttl": 7200
}
```

**Example Response**:
```json
{
  "enabled": true,
  "max_size": 2000,
  "ttl_seconds": 7200,
  "entries": 45,
  "hit_rate": 0.672
}
```

- **URL**: `/cache/clear`
- **Method**: `POST`
- **Description**: Clears the prediction cache.
- **Response**: Status message indicating success.

**Example Response**:
```json
{
  "status": "success",
  "message": "Cache cleared successfully"
}
```

### Prediction

- **URL**: `/predict`
- **Method**: `POST`
- **Description**: Submits patient data for heart disease risk prediction.
- **Request**: JSON object containing patient medical parameters.
- **Response**: Prediction results including the classification, probability, and risk level interpretation.

## Request and Response Formats

### Prediction Request

The prediction endpoint expects a JSON object with the following fields:

| Field | Type | Description | Required | Range |
|-------|------|-------------|----------|-------|
| age | integer | Age in years | Yes | 20-100 |
| sex | integer | Gender (1=male, 0=female) | Yes | 0-1 |
| cp | integer | Chest pain type | Yes | 0-3 |
| trestbps | integer | Resting blood pressure (mm Hg) | Yes | 90-200 |
| chol | integer | Serum cholesterol (mg/dl) | Yes | 100-600 |
| fbs | integer | Fasting blood sugar > 120 mg/dl (1=true, 0=false) | Yes | 0-1 |
| restecg | integer | Resting ECG results | Yes | 0-2 |
| thalach | integer | Maximum heart rate achieved | Yes | 60-220 |
| exang | integer | Exercise induced angina (1=yes, 0=no) | Yes | 0-1 |
| oldpeak | float | ST depression induced by exercise relative to rest | Yes | 0.0-10.0 |
| slope | integer | Slope of the peak exercise ST segment | Yes | 0-2 |
| ca | integer | Number of major vessels colored by fluoroscopy | Yes | 0-4 |
| thal | integer | Thalassemia type | Yes | 0-3 |

See the [Data Dictionary](data_dictionary.md) for detailed descriptions of each field.

**Example Request**:
```json
{
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
}
```

### Prediction Response

The prediction endpoint returns a JSON object with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| prediction | integer | Binary prediction (1=heart disease, 0=no heart disease) |
| probability | float | Probability of heart disease (0.0 to 1.0) |
| risk_level | string | Interpreted risk level ("LOW", "MODERATE", or "HIGH") |
| model_used | string | The model used for prediction (e.g., "ensemble", "sklearn_mlp", "keras_mlp") |
| interpretation | string | Clinical interpretation of the results (if requested) |
| feature_importances | object | Contribution of each feature to the prediction (if requested) |
| timestamp | string | ISO format timestamp of when the prediction was made |

**Example Response**:
```json
{
  "prediction": 1,
  "probability": 0.87,
  "risk_level": "HIGH",
  "model_used": "ensemble",
  "interpretation": "The patient has a high risk of heart disease. Key risk factors include exercise-induced angina, abnormal resting ECG, and elevated ST depression.",
  "feature_importances": {
    "age": 0.12,
    "sex": 0.09,
    "cp": 0.28,
    "trestbps": 0.06,
    "chol": 0.08,
    "fbs": 0.02,
    "restecg": 0.11,
    "thalach": 0.15,
    "exang": 0.22,
    "oldpeak": 0.19,
    "slope": 0.09,
    "ca": 0.18,
    "thal": 0.14
  },
  "timestamp": "2023-10-23T14:56:32.123Z"
}
```

### Batch Prediction Response

The batch prediction endpoint returns a JSON object with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| predictions | array | Array of prediction results for each patient |
| performance_metrics | object | Performance metrics for the batch operation (if enabled) |

Each prediction in the array follows the same format as the single prediction response. The performance_metrics object includes:

| Field | Type | Description |
|-------|------|-------------|
| total_patients | integer | Total number of patients processed |
| processing_time_seconds | float | Total processing time in seconds |
| throughput_patients_per_second | float | Processing throughput rate |
| num_chunks | integer | Number of chunks the batch was divided into |
| chunk_size | integer | Size of each processing chunk |
| num_workers | integer | Number of parallel workers used |

**Example Response**:
```json
{
  "predictions": [
    {
      "prediction": 1,
      "probability": 0.87,
      "risk_level": "HIGH",
      "model_used": "ensemble",
      "interpretation": "The patient has a high risk of heart disease."
    },
    {
      "prediction": 0,
      "probability": 0.12,
      "risk_level": "LOW",
      "model_used": "ensemble",
      "interpretation": "The patient has a low risk of heart disease."
    }
  ],
  "performance_metrics": {
    "total_patients": 2,
    "processing_time_seconds": 0.156,
    "throughput_patients_per_second": 12.82,
    "num_chunks": 1,
    "chunk_size": 50,
    "num_workers": 4
  }
}
```

## Error Handling

The API uses standard HTTP status codes to indicate the success or failure of a request:

- `200 OK`: Request successful.
- `400 Bad Request`: Invalid input data (e.g., missing required fields, values out of range).
- `404 Not Found`: The requested resource does not exist.
- `422 Unprocessable Entity`: Request validation error.
- `500 Internal Server Error`: An error occurred on the server.

Error responses include a JSON object with details:

```json
{
  "error": true,
  "message": "Detailed error message",
  "code": "ERROR_CODE",
  "details": {
    "field": "Additional information about the error"
  }
}
```

## Rate Limiting

The API implements rate limiting to prevent abuse:

- 100 requests per minute per IP address for the prediction endpoint
- 200 requests per minute per IP address for other endpoints

Rate limit headers are included in the response:

- `X-RateLimit-Limit`: Maximum requests per minute
- `X-RateLimit-Remaining`: Remaining requests in the current time window
- `X-RateLimit-Reset`: Seconds until the rate limit resets

## Example Usage

### Using cURL

```bash
# Health check
curl -X GET http://localhost:8000/health

# Model information
curl -X GET http://localhost:8000/models/info

# Get batch configuration
curl -X GET http://localhost:8000/batch/config

# Update batch configuration
curl -X POST http://localhost:8000/batch/config \
  -H "Content-Type: application/json" \
  -d '{
    "batch_size": 100,
    "max_workers": 8,
    "performance_logging": true
  }'

# Get cache statistics
curl -X GET http://localhost:8000/cache/stats

# Update cache configuration
curl -X POST http://localhost:8000/cache/config \
  -H "Content-Type: application/json" \
  -d '{
    "enabled": true,
    "max_size": 2000,
    "ttl": 7200
  }'

# Clear cache
curl -X POST http://localhost:8000/cache/clear

# Make a prediction
curl -X POST http://localhost:8000/predict \
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
  }'

# Make a batch prediction
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '[
    {
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
    },
    {
      "age": 45,
      "sex": 0,
      "cp": 1,
      "trestbps": 120,
      "chol": 180,
      "fbs": 0,
      "restecg": 0,
      "thalach": 175,
      "exang": 0,
      "oldpeak": 0.2,
      "slope": 1,
      "ca": 0,
      "thal": 2
    }
  ]'
```

### Using Python

```python
import requests
import json

# Base URL
base_url = "http://localhost:8000"

# Health check
response = requests.get(f"{base_url}/health")
print(f"Health check: {response.json()}")

# Model information
response = requests.get(f"{base_url}/models/info")
print(f"Model information: {response.json()}")

# Get batch configuration
response = requests.get(f"{base_url}/batch/config")
print(f"Batch configuration: {response.json()}")

# Update batch configuration for performance
batch_config = {
    "batch_size": 100,
    "max_workers": 8,
    "performance_logging": True
}
response = requests.post(f"{base_url}/batch/config", json=batch_config)
print(f"Updated batch configuration: {response.json()}")

# Make a single prediction
patient_data = {
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
}

response = requests.post(f"{base_url}/predict", json=patient_data)

if response.status_code == 200:
    result = response.json()
    print(f"Prediction: {result['prediction']}")
    print(f"Probability: {result['probability']}")
    print(f"Risk Level: {result['risk_level']}")
else:
    print(f"Error: {response.status_code} - {response.text}")

# Make a batch prediction
batch_data = [
    {
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
    },
    {
        "age": 45,
        "sex": 0,
        "cp": 1,
        "trestbps": 120,
        "chol": 180,
        "fbs": 0,
        "restecg": 0,
        "thalach": 175,
        "exang": 0,
        "oldpeak": 0.2,
        "slope": 1,
        "ca": 0,
        "thal": 2
    }
]

response = requests.post(f"{base_url}/predict/batch", json=batch_data)

if response.status_code == 200:
    result = response.json()
    print(f"Processed {len(result['predictions'])} patients")

    # Print predictions
    for i, prediction in enumerate(result['predictions']):
        print(f"Patient {i+1}:")
        print(f"  Prediction: {prediction['prediction']}")
        print(f"  Probability: {prediction['probability']}")
        print(f"  Risk Level: {prediction['risk_level']}")

    # Print performance metrics if available
    if 'performance_metrics' in result:
        metrics = result['performance_metrics']
        print("\nPerformance Metrics:")
        print(f"  Processing Time: {metrics['processing_time_seconds']} seconds")
        print(f"  Throughput: {metrics['throughput_patients_per_second']} patients/second")
        print(f"  Chunks: {metrics['num_chunks']} (size: {metrics['chunk_size']})")
        print(f"  Workers: {metrics['num_workers']}")
else:
    print(f"Error: {response.status_code} - {response.text}")
```

### Using JavaScript

```javascript
// Base URL
const baseUrl = "http://localhost:8000";

// Health check
fetch(`${baseUrl}/health`)
  .then(response => response.json())
  .then(data => console.log("Health check:", data))
  .catch(error => console.error("Error:", error));

// Model information
fetch(`${baseUrl}/models/info`)
  .then(response => response.json())
  .then(data => console.log("Model information:", data))
  .catch(error => console.error("Error:", error));

// Make a prediction
const patientData = {
  age: 61,
  sex: 1,
  cp: 3,
  trestbps: 140,
  chol: 240,
  fbs: 1,
  restecg: 1,
  thalach: 150,
  exang: 1,
  oldpeak: 2.4,
  slope: 2,
  ca: 1,
  thal: 3
};

fetch(`${baseUrl}/predict`, {
  method: "POST",
  headers: {
    "Content-Type": "application/json"
  },
  body: JSON.stringify(patientData)
})
  .then(response => response.json())
  .then(result => {
    console.log("Prediction:", result.prediction);
    console.log("Probability:", result.probability);
    console.log("Risk Level:", result.risk_level);
  })
  .catch(error => console.error("Error:", error));
```

## API Testing Script

The project includes a script for testing the API endpoints. Run it from the project root:

```bash
./scripts/test_api.sh
```

The script tests:
1. Health check endpoint
2. Model information endpoint
3. Prediction endpoint with sample data
4. Error handling for invalid input data

You can also refer to the integration tests in `tests/test_api.py` for additional examples.
