# Heart Disease Prediction System
A comprehensive machine learning system for predicting heart disease risk based on clinical parameters, using Multi-Layer Perceptron (MLP) neural networks.

![System Architecture](docs/figures/system_architecture.png)

## Table of Contents

- [Features](#features)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
  - [Data Processing](#data-processing)
  - [Model Training](#model-training)
  - [Making Predictions](#making-predictions)
  - [API Service](#api-service)
  - [Docker Deployment](#docker-deployment)
- [Interactive Tutorial](#interactive-tutorial)
- [Model Architecture](#model-architecture)
- [Performance Metrics](#performance-metrics)
- [CI/CD Pipeline](#cicd-pipeline)
- [Documentation](#documentation)
- [Security Features](#security-features)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features

- **Multiple model implementations**:
  - Scikit-learn MLP Classifier
  - Keras/TensorFlow deep learning model
  - Ensemble model combining both approaches

- **FastAPI REST API**:
  - Single prediction endpoint
  - Batch processing capability
  - Authentication with JWT and API keys
  - Swagger/OpenAPI documentation

- **Performance optimizations**:
  - Caching mechanism for repeated predictions
  - Parallel processing for batch requests
  - Configurable batch size and worker count

- **Comprehensive testing**:
  - Unit tests for all components
  - Integration and performance tests
  - Error handling tests
  - Authentication and security tests

- **Production-ready features**:
  - Docker containerization
  - Environment-specific configuration
  - Comprehensive backup and recovery
  - Monitoring and logging

## System Architecture

The Heart Disease Prediction System follows a modular architecture:

```
                  ┌──────────────┐
                  │  Web Client  │
                  └──────────────┘
                         │
                         ▼
┌─────────────────────────────────────────┐
│               FastAPI API               │
├─────────────────────────────────────────┤
│  ┌─────────────┐    ┌────────────────┐  │
│  │ Prediction  │    │ Authentication │  │
│  │  Endpoints  │    │    Service     │  │
│  └─────────────┘    └────────────────┘  │
│  ┌─────────────┐    ┌────────────────┐  │
│  │   Caching   │    │  Batch Process │  │
│  │   Service   │    │     Service    │  │
│  └─────────────┘    └────────────────┘  │
└─────────────────────────────────────────┘
                 │        │
        ┌────────┘        └────────┐
        ▼                          ▼
┌─────────────────┐      ┌──────────────────┐
│ Scikit-learn    │      │ Keras/TensorFlow │
│  MLP Model      │      │   Neural Network │
└─────────────────┘      └──────────────────┘
        │                          │
        └────────────┬─────────────┘
                     ▼
           ┌───────────────────┐
           │ Ensemble Combiner │
           └───────────────────┘
                     │
                     ▼
             ┌──────────────┐
             │ Preprocessor │
             └──────────────┘
```

## Project Structure

```
heart-disease-mlp/
├── .github/                # GitHub workflows for CI/CD
│   └── workflows/
├── api/                    # FastAPI application
│   ├── app.py              # Main API module
│   ├── auth.py             # Authentication functions
│   └── static/             # Static web files
├── backups/                # Backup storage location
├── config/                 # Configuration files
│   ├── config.yaml         # Default configuration
│   ├── config.dev.yaml     # Development configuration
│   ├── config.staging.yaml # Staging configuration
│   └── config.prod.yaml    # Production configuration
├── data/                   # Data directory
│   ├── examples/           # Example input data
│   ├── external/           # External data sources
│   ├── processed/          # Processed data
│   └── raw/                # Raw data
├── docs/                   # Documentation
│   ├── README.md           # Documentation index
│   ├── api.md              # API documentation
│   ├── api_usage_examples.md # API usage examples
│   ├── backup_recovery.md  # Backup & recovery guide
│   ├── cicd_status.md      # CI/CD pipeline status
│   ├── data_dictionary.md  # Data fields explanation
│   ├── environment_config.md # Environment configuration
│   ├── fixes_documentation.md # System improvements and bug fixes
│   ├── model.md            # Model architecture details
│   ├── system_architecture.md # System components architecture
│   └── usage.md            # Detailed usage instructions
├── models/                 # Trained models
│   ├── evaluation_results.joblib  # Performance metrics
│   ├── keras_mlp_model.h5         # Keras model weights
│   ├── optuna/                    # Hyperparameter tuning results
│   └── sklearn_mlp_model.joblib   # Scikit-learn model
├── notebooks/              # Jupyter notebooks for exploration
│   ├── heart_disease_prediction_tutorial.ipynb        # Original tutorial notebook (deprecated)
│   ├── heart_disease_prediction_tutorial_updated.ipynb # Updated tutorial (deprecated)
│   ├── heart_disease_prediction_tutorial_working.ipynb # Production-ready tutorial with robust error handling
│   └── test_notebook.ipynb                            # Test notebook for verifying dependencies
├── reports/                # Generated reports and visualizations
│   └── figures/            # Performance visualizations
├── scripts/                # Shell scripts for automation
│   ├── backup_system.py    # Backup and recovery system
│   ├── deploy_api.sh       # API deployment script
│   ├── get_data.sh         # Data download script
│   ├── process_data.sh     # Data processing script
│   ├── train_models.sh     # Model training script
│   └── run_api.sh          # API startup script
├── src/                    # Source code
│   ├── data/               # Data processing modules
│   │   ├── download_data.py # Data acquisition
│   │   ├── make_dataset.py # Dataset creation
│   │   └── preprocess.py   # Data preprocessing
│   ├── features/           # Feature engineering
│   │   └── feature_engineering.py # Feature creation/selection
│   ├── models/             # Model implementation
│   │   ├── hyperparameter_tuning.py # Hyperparameter optimization
│   │   ├── mlp_model.py    # MLP model implementation
│   │   ├── predict_model.py # Prediction functions
│   │   └── train_model.py  # Model training
│   └── visualization/      # Visualization code
│       └── visualize.py    # Plotting functions
├── tests/                  # Test suite
│   ├── conftest.py         # Test configuration
│   ├── test_api.py         # API tests
│   ├── test_auth.py        # Authentication tests
│   ├── test_cache.py       # Caching tests
│   ├── test_data.py        # Data processing tests
│   ├── test_error_handling.py # Error handling tests
│   └── test_models.py      # Model functionality tests
├── .gitignore              # Git ignore file
├── Dockerfile              # Docker configuration
├── docker-compose.yaml     # Docker Compose configuration
├── docker-compose.dev.yaml # Development Docker Compose configuration
├── docker-compose.staging.yaml # Staging Docker Compose configuration
├── docker-compose.prod.yaml # Production Docker Compose configuration
├── README.md               # Project README
├── requirements.txt        # Python dependencies
└── setup.py                # Package setup script
```

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/heart-disease-mlp.git
   cd heart-disease-mlp
   ```

2. **Create and activate a virtual environment:**

   ```bash
   # Create a virtual environment
   python -m venv venv

   # Activate it
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   # Install project dependencies
   pip install -r requirements.txt

   # Install the package in development mode
   pip install -e .
   ```

4. **Verify installation:**

   ```bash
   # Run tests to ensure everything is working
   pytest tests/

   # Verify tutorial notebook dependencies
   jupyter nbconvert --to html --execute notebooks/test_notebook.ipynb
   
   # Run the full tutorial notebook
   jupyter nbconvert --to html --execute notebooks/heart_disease_prediction_tutorial_working.ipynb
   ```

## Usage Guide

### Data Processing

1. **Download and prepare the dataset:**

   ```bash
   # Download the heart disease dataset
   ./scripts/get_data.sh

   # Process the raw data
   ./scripts/process_data.sh
   ```

   This will:
   - Download the heart disease dataset from multiple sources
   - Clean and preprocess the data
   - Create train/validation/test splits
   - Save the processed data to `data/processed/`

### Model Training

2. **Train the models:**

   ```bash
   # Train all models (scikit-learn MLP, Keras MLP, and ensemble)
   ./scripts/train_models.sh
   ```

   This will:
   - Train a scikit-learn MLP classifier
   - Train a Keras/TensorFlow neural network
   - Create an ensemble model
   - Save models to the `models/` directory
   - Generate performance reports in `reports/figures/`

### Running the API

3. **Start the prediction API:**

   ```bash
   # Start the API server
   ./scripts/run_api.sh

   # Alternatively, use Docker Compose:
   docker-compose up -d
   ```

   The API will be available at:
   - http://localhost:8000/ (Web Interface)
   - http://localhost:8000/docs (Swagger UI)
   - http://localhost:8000/redoc (ReDoc)

### Making Predictions

4. **Make predictions using the API:**

   ```bash
   # Example using curl for a single prediction
   curl -X POST "http://localhost:8000/predict" \
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

   See `docs/api_usage_examples.md` for more examples including batch prediction.

### Backup and Recovery

5. **Create a backup of models and data:**

   ```bash
   # Create a backup
   ./scripts/backup.sh create

   # List available backups
   ./scripts/backup.sh list

   # Restore from a backup
   ./scripts/backup.sh restore <backup_id>
   ```

## Documentation

- **API Usage:** See [API Documentation](docs/api.md)
- **Model Details:** See [Model Architecture](docs/model.md)
- **Data Fields:** See [Data Dictionary](docs/data_dictionary.md)
- **Configuration:** See [Environment Configuration](docs/environment_config.md)
- **Backup & Recovery:** See [Backup and Recovery](docs/backup_recovery.md)

## Interactive Tutorial

For an interactive tutorial, run the Jupyter notebook:

```bash
jupyter notebook notebooks/heart_disease_prediction_tutorial_working.ipynb
```

The tutorial covers:
- Data exploration and preprocessing
- Model training and evaluation
- Making predictions with the API
- Batch processing and performance considerations
- Environment-specific configuration

## Model Architecture

The project implements two complementary MLP models based on the configuration in `config/config.yaml`:

1. **Scikit-learn MLP**:
   - Two hidden layers (100, 50 neurons)
   - ReLU activation function
   - Adam optimizer
   - L2 regularization (alpha=0.0001)
   - Maximum 2000 iterations

2. **Keras MLP**:
   - Three hidden layers with:
     - Layer 1: 128 units, LeakyReLU, 30% dropout, L2 regularization
     - Layer 2: 64 units, LeakyReLU, 30% dropout, L2 regularization
     - Layer 3: 32 units, LeakyReLU, 20% dropout, L2 regularization
   - Batch size of 32
   - Early stopping with patience of 20
   - Learning rate reduction with patience of 10
   - Adam optimizer with 0.001 learning rate

3. **Ensemble Model**:
   - Combines predictions from both models
   - Improves overall robustness and performance

For more details, see the [model documentation](docs/model.md).

## Performance Metrics

The models are evaluated using multiple metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- AUC-ROC

Performance visualizations are available in the reports directory:
- ROC curves: `reports/figures/roc_curve_*.png`
- Precision-Recall curves: `reports/figures/pr_curve_*.png`
- Confusion matrices: `reports/figures/confusion_matrix_*.png`

## CI/CD Pipeline

This project uses GitHub Actions (`.github/workflows/`) for continuous integration and deployment:

- **Code Quality**: Enforces consistent code formatting
- **Dependency Management**: Maintains up-to-date dependencies
- **Automated Testing**: Runs the test suite on code changes
- **Security Scanning**: Checks for vulnerabilities
- **Model Retraining**: Scheduled workflow for model retraining

## Security Features

- **Authentication:** JWT tokens and API keys
- **Authorization:** Role-based access control
- **Data Validation:** Pydantic schema validation
- **Error Handling:** Comprehensive exception management
- **Backup:** Automated backup and recovery procedures

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset source: UCI Machine Learning Repository
- Based on research from Cleveland Clinic Foundation and other institutions