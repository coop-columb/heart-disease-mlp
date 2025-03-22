# Heart Disease Prediction Model

This project implements a machine learning system for predicting heart disease risk based on clinical parameters using Multi-Layer Perceptron (MLP) neural networks.

## Overview

The system uses both scikit-learn and TensorFlow/Keras MLP models and combines them into an ensemble for improved prediction accuracy. The project follows a modular architecture and includes components for data processing, feature engineering, model training, evaluation, and deployment.

## Features

- **Data Processing Pipeline**: Automated data cleaning, preprocessing, and feature engineering
- **Multiple MLP Models**: Implementation of both scikit-learn and Keras MLPs with optimized architectures
- **Hyperparameter Tuning**: Optuna-based hyperparameter optimization
- **Model Ensembling**: Combined predictions from multiple models for improved accuracy
- **Interpretation**: Clinical interpretation of predictions
- **API**: FastAPI-based REST API for model serving
- **Containerization**: Docker setup for easy deployment
- **CI/CD Pipeline**: GitHub Actions workflows for automated testing, building, and deployment
- **Security Scanning**: Automated vulnerability scanning for dependencies and Docker images

## Project Structure

```
heart-disease-mlp/
├── .github/            # GitHub configuration
│   └── workflows/      # GitHub Actions CI/CD workflows
├── api/                # API implementation
├── config/             # Configuration files
├── data/               # Data storage
│   ├── external/       # External reference data
│   ├── processed/      # Processed datasets
│   └── raw/            # Raw datasets
├── models/             # Trained models
├── notebooks/          # Jupyter notebooks
├── reports/            # Generated reports and visualizations
├── scripts/            # Shell scripts for automation
└── src/                # Source code
    ├── data/           # Data processing modules
    ├── features/       # Feature engineering
    ├── models/         # Model implementation
    └── visualization/  # Visualization utilities
```

## Setup and Installation

### Requirements

- Python 3.8+
- Required libraries (see requirements.txt)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/heart-disease-mlp.git
   cd heart-disease-mlp
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Pipeline

1. Get and process the data:
   ```bash
   ./scripts/get_data.sh
   ./scripts/process_data.sh
   ```

2. Train the models:
   ```bash
   ./scripts/train_models.sh
   ```

3. For hyperparameter tuning:
   ```bash
   ./scripts/train_models.sh --tune
   ```

## Model Deployment

### Local API

Run the API server locally:

```bash
uvicorn api.app:app --reload
```

The API will be available at http://localhost:8000, with interactive documentation at http://localhost:8000/docs.

### Docker

Deploy using Docker:

```bash
docker-compose up -d
```

## Making Predictions

### Using the API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 65,
    "sex": 1,
    "cp": 0,
    "trestbps": 140,
    "chol": 220,
    "fbs": 1,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.2,
    "slope": 2,
    "ca": 0,
    "thal": 2
  }'
```

### Using the CLI

```bash
python src/models/cli_predict.py --input data/examples/patient_example.json --interpretation
```

## Model Performance

The current models achieve the following performance on the test set:

- Scikit-learn MLP: 85.3% accuracy, 0.929 ROC AUC
- Keras MLP: 85.3% accuracy, 0.926 ROC AUC
- Ensemble: 86.4% accuracy, 0.930 ROC AUC

## CI/CD Pipeline

This project uses GitHub Actions for continuous integration and deployment:

- **Automated Testing**: Runs unit tests and integration tests on multiple Python versions
- **Code Quality**: Enforces code style with black, isort, and flake8
- **Docker Image Building**: Builds and pushes Docker images to GitHub Container Registry
- **Automatic Deployment**: Deploys to staging and production environments
- **Security Scanning**: Checks dependencies and Docker images for vulnerabilities
- **Automated Model Retraining**: Schedules monthly model retraining and evaluation

For more details, see the [workflows documentation](.github/workflows/README.md).

## License

This project is licensed under the MIT License - see the LICENSE file for details.
