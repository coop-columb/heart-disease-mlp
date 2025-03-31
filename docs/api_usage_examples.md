# Heart Disease Prediction API Usage Examples

This document provides examples of how to use the Heart Disease Prediction API with various programming languages and tools.

## API Endpoints

The API provides the following endpoints:

- `GET /health` - Check the health of the API
- `GET /models/info` - Get information about available models
- `POST /predict` - Make a prediction for a single patient
- `POST /predict/batch` - Make predictions for multiple patients
- `GET /` - Access the web UI

## cURL Examples

### Health Check

Check if the API is running:

```bash
curl -X GET http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy"
}
```

### Get Models Information

Get information about available models:

```bash
curl -X GET http://localhost:8000/models/info
```

Expected response:
```json
{
  "models_available": {
    "sklearn_mlp": true,
    "keras_mlp": true
  },
  "ensemble_available": true,
  "preprocessor_available": true
}
```

### Make a Prediction (Default Ensemble Model)

Make a prediction using the default ensemble model:

```bash
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
```

### Make a Prediction (Specific Model)

Make a prediction using a specific model (sklearn or keras):

```bash
curl -X POST "http://localhost:8000/predict?model=sklearn" \
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
```

### Batch Prediction

Make predictions for multiple patients at once:

```bash
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

## Python Examples

### Basic Request

Here's a simple Python script to make a prediction:

```python
import requests
import json

# API endpoint
url = "http://localhost:8000/predict"

# Patient data
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

# Make the request
response = requests.post(url, json=patient_data)

# Print the results
if response.status_code == 200:
    result = response.json()
    print("Prediction:", "Heart Disease" if result["prediction"] == 1 else "No Heart Disease")
    print(f"Probability: {result['probability']:.2f}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Model Used: {result['model_used']}")
    print("\nInterpretation:")
    print(result["interpretation"])
else:
    print(f"Error: {response.status_code}")
    print(response.json())
```

### Batch Prediction

Here's how to make batch predictions in Python:

```python
import requests
import json

# API endpoint
url = "http://localhost:8000/predict/batch"

# Batch of patient data
patients = [
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

# Make the request
response = requests.post(url, json=patients)

# Print the results
if response.status_code == 200:
    results = response.json()["predictions"]
    for i, result in enumerate(results):
        print(f"\nPatient {i+1}:")
        print("Prediction:", "Heart Disease" if result["prediction"] == 1 else "No Heart Disease")
        print(f"Probability: {result['probability']:.2f}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Model Used: {result['model_used']}")
else:
    print(f"Error: {response.status_code}")
    print(response.json())
```

## JavaScript (Node.js) Example

```javascript
const fetch = require('node-fetch');

// API endpoint
const url = 'http://localhost:8000/predict';

// Patient data
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

// Make the request
fetch(url, {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify(patientData),
})
.then(response => response.json())
.then(result => {
    console.log('Prediction:', result.prediction === 1 ? 'Heart Disease' : 'No Heart Disease');
    console.log(`Probability: ${result.probability.toFixed(2)}`);
    console.log(`Risk Level: ${result.risk_level}`);
    console.log(`Model Used: ${result.model_used}`);
    console.log('\nInterpretation:');
    console.log(result.interpretation);
})
.catch(error => {
    console.error('Error:', error);
});
```

## Integration with Electronic Health Records

For integration with EHR systems, we recommend creating a middleware service that:

1. Extracts relevant patient data from your EHR system
2. Transforms it to match the API input format
3. Makes API calls to the prediction service
4. Stores or displays the results appropriately

This can be done using any programming language that supports HTTP requests, based on your EHR system's technology stack.

## Load Testing

For load testing the API, you can use tools like Apache JMeter, Locust, or k6. Here's an example script for k6:

```javascript
import http from 'k6/http';
import { sleep, check } from 'k6';

export const options = {
    vus: 10,            // Number of virtual users
    duration: '30s',    // Test duration
};

export default function () {
    // Patient data
    const payload = JSON.stringify({
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
    });

    // Make the request
    const res = http.post('http://localhost:8000/predict', payload, {
        headers: { 'Content-Type': 'application/json' },
    });

    // Check if the request was successful
    check(res, {
        'status is 200': (r) => r.status === 200,
        'has prediction': (r) => JSON.parse(r.body).prediction !== undefined,
    });

    // Wait before making the next request
    sleep(1);
}
```

## Error Handling

The API implements robust error handling. Here are some examples of how errors are handled:

### Missing Data Field

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 61,
    "sex": 1
  }'
```

The API will return a 422 Unprocessable Entity response with details about the missing fields.

### Model Not Available

If you request a specific model that is not available:

```bash
curl -X POST "http://localhost:8000/predict?model=xgboost" \
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
```

The API will use a fallback model and include information about which model was used in the response.

## Performance Considerations

- The API is designed to handle multiple concurrent requests
- For batch predictions, consider limiting the batch size to 100 patients for optimal performance
- The API has a response time of ~50ms for single predictions on modest hardware
- For high-volume production use, consider deploying behind a load balancer

## Security Recommendations

When deploying to production:

1. Add authentication to the API using JWT or API keys
2. Use HTTPS to encrypt data in transit
3. Implement rate limiting to prevent abuse
4. Consider adding IP restrictions or VPN requirements for sensitive deployments
5. Regularly update dependencies to address security vulnerabilities