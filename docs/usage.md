# Heart Disease Prediction System: Usage Guide

This comprehensive guide explains how to use the Heart Disease Prediction System for various use cases, from data processing to deployment and monitoring.

## Table of Contents

- [Project Setup](#project-setup)
- [Data Processing Workflow](#data-processing-workflow)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Making Predictions](#making-predictions)
- [API Deployment](#api-deployment)
- [Docker Containerization](#docker-containerization)
- [CI/CD Integration](#cicd-integration)
- [Monitoring and Logging](#monitoring-and-logging)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)
- [Contributing](#contributing)

## Project Setup

### Initial Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/heart-disease-mlp.git
   cd heart-disease-mlp
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # Using venv
   python -m venv venv

   # On Unix/MacOS
   source venv/bin/activate

   # On Windows
   venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt

   # Development installation (editable mode)
   pip install -e .
   ```

4. **Verify installation**:
   ```bash
   # Run tests to ensure everything works correctly
   pytest tests/
   ```

### Configuration

The project uses YAML configuration files located in the `config/` directory:

- **Main configuration file**: `config/config.yaml` contains settings for:
  - Data processing parameters
  - Model training parameters
  - API settings
  - Evaluation metrics

To customize the configuration:

1. Create a copy of the default configuration:
   ```bash
   cp config/config.yaml config/config_custom.yaml
   ```

2. Edit `config_custom.yaml` with your preferred settings.

3. Specify the custom configuration when running scripts:
   ```bash
   # Example: Using custom config for training
   python -m src.models.train_model --config config/config_custom.yaml
   ```

## Data Processing Workflow

### Downloading Data

1. **Automatic download using script**:
   ```bash
   ./scripts/get_data.sh
   ```
   This downloads heart disease datasets from the UCI Machine Learning Repository and saves them to `data/raw/`.

2. **Manual download**:
   ```bash
   # Run the Python module directly
   python -m src.data.download_data

   # With custom output directory
   python -m src.data.download_data --output-dir data/custom_raw/
   ```

### Data Preprocessing

1. **Process the data using script**:
   ```bash
   ./scripts/process_data.sh
   ```
   This runs the preprocessing pipeline, which includes:
   - Data cleaning
   - Feature engineering
   - Feature scaling
   - Train/validation/test splitting

2. **Manual processing with options**:
   ```bash
   # Run with custom parameters
   python -m src.data.make_dataset --raw-dir data/raw/ --processed-dir data/processed/ --test-size 0.2 --val-size 0.15 --random-state 42
   ```

3. **Processing outputs**:
   - `data/processed/processed_data.npz`: Processed data arrays
   - `data/processed/preprocessor.joblib`: Serialized preprocessor object
   - `data/processed/original_splits.joblib`: Train/val/test splits indices
   - `data/processed/processing_metadata.txt`: Information about the processing steps

### Feature Engineering

To customize feature engineering:

1. Edit `src/features/feature_engineering.py` to add new features.

2. Run the preprocessing pipeline again:
   ```bash
   python -m src.data.make_dataset --reprocess
   ```

## Model Training and Evaluation

### Training Models

1. **Train all models using script**:
   ```bash
   ./scripts/train_models.sh
   ```
   This trains both scikit-learn and Keras MLPs with default parameters.

2. **Train with hyperparameter tuning**:
   ```bash
   ./scripts/train_models.sh --tune
   ```
   This runs Optuna hyperparameter optimization before training.

3. **Manual training with options**:
   ```bash
   # Train only scikit-learn MLP
   python -m src.models.train_model --model sklearn --data-path data/processed/processed_data.npz

   # Train only Keras MLP with custom parameters
   python -m src.models.train_model --model keras --data-path data/processed/processed_data.npz --epochs 150 --batch-size 32 --patience 50
   ```

### Hyperparameter Tuning

1. **Run hyperparameter tuning manually**:
   ```bash
   python -m src.models.hyperparameter_tuning --model sklearn --n-trials 100 --study-name sklearn_mlp_study
   ```

2. **Visualize tuning results**:
   ```bash
   python -m src.visualization.visualize --plot-optuna --study-name sklearn_mlp_study
   ```

### Model Evaluation

1. **Evaluate trained models**:
   ```bash
   python -m src.models.evaluate_model
   ```
   This generates:
   - Performance metrics (accuracy, precision, recall, F1, ROC AUC)
   - Confusion matrices
   - ROC curves
   - Precision-Recall curves
   - Feature importance plots

2. **Custom evaluation**:
   ```bash
   python -m src.models.evaluate_model --model keras --data-split test --save-plots
   ```

3. **Visualize results**:
   ```bash
   # Generate all visualizations
   python -m src.visualization.visualize --all-plots

   # Generate specific plot
   python -m src.visualization.visualize --plot-roc --model ensemble
   ```

## Making Predictions

### Using the Command Line Interface (CLI)

1. **Predict from a JSON file**:
   ```bash
   python -m src.models.cli_predict --input data/examples/patient_example.json
   ```

2. **Predict with a specific model**:
   ```bash
   python -m src.models.cli_predict --input data/examples/patient_example.json --model keras
   ```

3. **Get detailed interpretation**:
   ```bash
   python -m src.models.cli_predict --input data/examples/patient_example.json --interpretation
   ```

4. **Output to file**:
   ```bash
   python -m src.models.cli_predict --input data/examples/patient_example.json --output predictions.json
   ```

5. **Batch prediction**:
   ```bash
   python -m src.models.cli_predict --input data/examples/patients_batch.json --batch
   ```

### Input Format

Create a JSON file with patient data in the following format:

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

For batch predictions, use an array of objects:

```json
[
  { "age": 61, "sex": 1, ... },
  { "age": 45, "sex": 0, ... },
  ...
]
```

## API Deployment

### Local Deployment

1. **Run the API server locally**:
   ```bash
   # Basic usage
   uvicorn api.app:app --reload

   # Specify host and port
   uvicorn api.app:app --host 0.0.0.0 --port 8080 --reload
   ```

2. **Deploy using script**:
   ```bash
   ./scripts/deploy_api.sh
   ```

3. **Test the API**:
   ```bash
   ./scripts/test_api.sh
   ```

### API Usage

1. **Access API documentation**:
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

2. **Health check**:
   ```bash
   curl -X GET http://localhost:8000/health
   ```

3. **Get model information**:
   ```bash
   curl -X GET http://localhost:8000/models/info
   ```

4. **Make a prediction**:
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

## Docker Containerization

### Building and Running with Docker

1. **Build the Docker image**:
   ```bash
   docker build -t heart-disease-prediction .
   ```

2. **Run the container**:
   ```bash
   docker run -p 8000:8000 heart-disease-prediction
   ```

3. **Using Docker Compose**:
   ```bash
   # Start services
   docker-compose up -d

   # Stop services
   docker-compose down

   # View logs
   docker-compose logs -f
   ```

### Docker Compose Configuration

The `docker-compose.yml` file defines the following services:

1. **API Service**:
   - Based on the project's Dockerfile
   - Exposes port 8000
   - Mounts volumes for models and data

2. **Volumes**:
   - `models`: Persists trained models
   - `data`: Stores datasets

### Custom Docker Settings

To customize Docker deployment:

1. Edit environment variables in `docker-compose.yml`:
   ```yaml
   environment:
     - MODEL_DIR=/app/models
     - LOG_LEVEL=INFO
     - PORT=8000
   ```

2. Modify resource constraints:
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '1'
         memory: 2G
   ```

## CI/CD Integration

### GitHub Actions Setup

1. **Required Secrets**:
   Set up the following secrets in your GitHub repository:
   - `SSH_PRIVATE_KEY`: SSH key for deployment
   - `DEPLOY_HOST`: Hostname for staging server
   - `DEPLOY_USER`: Username for staging server
   - `PROD_DEPLOY_HOST`: Hostname for production server
   - `PROD_DEPLOY_USER`: Username for production server

2. **GitHub Environments**:
   Set up environments in your GitHub repository:
   - `staging`: For staging deployments
   - `production`: For production deployments (with required approvals)

For detailed setup instructions, see [CI/CD Setup Guide](.github/CICD_SETUP.md).

### Manual Workflow Triggers

1. **Trigger CI/CD pipeline**:
   - Go to Actions tab in GitHub
   - Select "CI/CD Pipeline" workflow
   - Click "Run workflow"

2. **Trigger model retraining**:
   - Go to Actions tab in GitHub
   - Select "Model Retraining and Evaluation" workflow
   - Click "Run workflow"

## Monitoring and Logging

### API Logging

1. **Configure log level**:
   - Edit `config/config.yaml` to set log level:
     ```yaml
     api:
       log_level: INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
     ```

2. **View logs**:
   ```bash
   # For local deployment
   tail -f api.log

   # For Docker deployment
   docker-compose logs -f
   ```

### Model Monitoring

1. **Performance monitoring**:
   The API tracks prediction statistics that can be accessed at:
   ```bash
   curl -X GET http://localhost:8000/stats
   ```

2. **Model drift detection**:
   Run the drift detection script periodically:
   ```bash
   python -m src.models.drift_detection --reference-data data/processed/reference_data.npz --current-data new_data.npz
   ```

## Troubleshooting

### Common Issues

1. **Missing Models**:
   If you get errors about missing models:
   ```bash
   # Train the models
   ./scripts/train_models.sh
   ```

2. **API Connection Issues**:
   - Check that the API server is running
   - Verify the host and port settings
   - Check firewall settings

3. **Docker Issues**:
   - Ensure Docker is running
   - Check available disk space
   - Verify port mappings are correct

### Error Logs

Check the logs for detailed error information:

```bash
# API logs
cat api.log

# Model training logs
cat logs/training.log

# Docker logs
docker-compose logs
```

## Advanced Usage

### Custom Model Architectures

To implement a custom model:

1. Create a new model definition in `src/models/mlp_model.py`:
   ```python
   def build_custom_model(input_dim, **kwargs):
       # Define your custom model
       # ...
       return model
   ```

2. Update the training script to include your model:
   ```python
   # In src/models/train_model.py
   if model_type == 'custom':
       model = build_custom_model(X_train.shape[1], **model_params)
   ```

### Model Interpretability

For detailed model interpretation:

```bash
# Generate SHAP values for model interpretability
python -m src.models.model_interpret --model ensemble --sample-data data/examples/interpretation_sample.json
```

### Custom Metrics

To add custom evaluation metrics:

1. Define the metric in `src/models/evaluate_model.py`:
   ```python
   def custom_metric(y_true, y_pred):
       # Implement your metric
       return score
   ```

2. Add the metric to the evaluation function:
   ```python
   metrics['custom'] = custom_metric(y_test, y_pred)
   ```

## Contributing

To contribute to this project:

1. **Setup development environment**:
   ```bash
   # Install development dependencies
   pip install -e ".[dev]"

   # Install pre-commit hooks
   pre-commit install
   ```

2. **Run tests before submitting changes**:
   ```bash
   # Run all tests
   pytest

   # Run with coverage
   pytest --cov=src
   ```

3. **Code style**:
   - Format code with Black: `black src tests api`
   - Sort imports with isort: `isort src tests api`
   - Check style with flake8: `flake8 src tests api`

For more details, see the contribution guidelines in the repository.
