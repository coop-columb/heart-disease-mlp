# Heart Disease Prediction System Architecture

This document provides a detailed overview of the Heart Disease Prediction system's architecture, highlighting its key components, data flow, and interactions.

## System Overview

The Heart Disease Prediction system employs a layered architecture that separates concerns and promotes maintainability:

1. **Data Layer**: Handles data acquisition, preprocessing, and feature engineering
2. **Model Layer**: Implements multiple machine learning models and prediction logic
3. **API Layer**: Exposes prediction functionality through RESTful endpoints
4. **Client Layer**: Provides interfaces for interacting with the system

## Architecture Diagram

![Heart Disease Prediction System Architecture](../reports/figures/system_architecture.png)

## Layer Descriptions

### 1. Data Layer

**Responsibilities**:
- Data acquisition from various sources
- Data cleaning and validation
- Feature preprocessing and engineering
- Dataset splitting (train/validation/test)
- Storage of preprocessed data and artifacts

**Key Components**:
- `src/data/download_data.py`: Fetches raw data from sources
- `src/data/make_dataset.py`: Orchestrates the data preprocessing pipeline
- `src/data/preprocess.py`: Implements preprocessing transformations
- `src/features/feature_engineering.py`: Creates derived features and transformations
- `data/processed/preprocessor.joblib`: Serialized preprocessing pipeline

**Data Flow**:
1. Raw data is downloaded from external sources
2. Data is cleaned and preprocessed
3. Features are engineered and transformed
4. Data is split into training, validation, and test sets
5. Preprocessed data and artifacts are saved for model training and inference

### 2. Model Layer

**Responsibilities**:
- Model definition and training
- Hyperparameter optimization
- Model evaluation and metrics
- Prediction logic and ensemble methods
- Model serialization and loading
- Prediction caching for performance

**Key Components**:
- `src/models/mlp_model.py`: Defines neural network architectures
- `src/models/train_model.py`: Implements model training procedures
- `src/models/predict_model.py`: Handles prediction logic and caching
- `src/models/hyperparameter_tuning.py`: Optimizes model parameters
- `models/*.joblib` and `models/*.h5`: Serialized model files

**Major Classes**:
- `HeartDiseasePredictor`: Core prediction class that interfaces with trained models
- `PredictionCache`: LRU cache implementation for improved prediction throughput

**Model Features**:
- Multiple model types (scikit-learn MLP, Keras MLP)
- Ensemble prediction combining multiple models
- Prediction interpretation for clinical context
- LRU caching with configurable TTL for performance
- Fallback mechanisms for model unavailability

### 3. API Layer

**Responsibilities**:
- Expose prediction functionality through RESTful endpoints
- Validate request inputs
- Transform predictions into standardized responses
- Handle errors gracefully
- Optimize batch processing with parallelization
- Provide configuration endpoints
- Manage prediction caching

**Key Components**:
- `api/app.py`: FastAPI application with endpoint definitions
- `/api/static/*`: Static files for the web UI

**Endpoints**:
- `/predict`: Single patient prediction
- `/predict/batch`: Batch prediction for multiple patients
- `/models/info`: Information about available models
- `/health`: System health check
- `/batch/config`: Batch processing configuration
- `/cache/stats`: Cache statistics endpoint
- `/cache/config`: Cache configuration endpoint
- `/cache/clear`: Cache clearing endpoint

**Performance Features**:
- Thread-based parallelization for batch processing
- Configurable batch chunking for large datasets
- LRU prediction caching with configurable parameters
- Performance metrics tracking

### 4. Client Layer

**Responsibilities**:
- Provide interfaces for interacting with the API
- Demonstrate proper API usage
- Facilitate testing and validation
- Present results in a user-friendly manner

**Key Components**:
- `scripts/manual_api_test.py`: Test script for API endpoints
- Web UI (served through FastAPI's static files)
- `notebooks/heart_disease_prediction_tutorial.ipynb`: Interactive tutorial
- CLI prediction interface

### 5. Utilities (Shared)

**Responsibilities**:
- Provide common functionality used across layers
- Centralize configuration management
- Implement logging and error handling

**Key Components**:
- `src/utils.py`: Utility functions for configuration loading, logging setup, etc.
- `config/config.yaml`: Centralized configuration file

## Interactions and Data Flow

1. **Prediction Flow**:
   - Client sends request to API endpoint
   - API validates request data
   - API forwards request to the model layer
   - Model layer checks cache for existing predictions
   - If not in cache, model preprocesses data and makes prediction
   - Prediction is cached for future use
   - API formats response and returns to client

2. **Batch Processing Flow**:
   - Client sends batch request to API
   - API divides batch into chunks
   - Chunks are processed in parallel using thread pool
   - Each chunk checks cache and processes predictions
   - Results are aggregated and returned to client with performance metrics

3. **Cache Management Flow**:
   - Client can configure cache parameters (size, TTL)
   - Cache automatically evicts least recently used entries when full
   - Cache entries expire based on TTL settings
   - Cache statistics are available for monitoring

## Key Performance Features

The system implements several optimizations for high performance:

1. **Prediction Caching**:
   - LRU (Least Recently Used) caching strategy
   - Configurable cache size and TTL (Time-To-Live)
   - Hash-based cache keys for efficient lookups
   - Thread-safe implementation for concurrent access
   - Cache statistics tracking for monitoring

2. **Parallel Processing**:
   - Batch requests are divided into configurable chunks
   - Chunks are processed in parallel using a thread pool
   - Configurable number of worker threads
   - Performance metrics tracking for optimization

3. **API Efficiency**:
   - FastAPI for high-performance request handling
   - Pydantic models for efficient request validation
   - Graceful error handling and recovery
   - Clear separation of concerns for maintainability

## Security Considerations

1. **Input Validation**:
   - All API inputs are validated through Pydantic models
   - Type checking and range validation for clinical parameters

2. **Error Handling**:
   - Comprehensive error handling throughout the pipeline
   - Graceful degradation when models are unavailable
   - Appropriate HTTP status codes for different error types

3. **Planned Enhancements**:
   - API authentication and authorization
   - Role-based access control
   - Environment-specific configuration
   - Secure deployment procedures

## Deployment Architecture

The system is designed to be deployed in various environments:

1. **Local Development**:
   - Run directly with Python and uvicorn
   - Configuration via local config file

2. **Docker Container**:
   - Containerized deployment with Docker
   - Multi-stage build for smaller image size
   - Environment variables for configuration

3. **Orchestrated Deployment** (planned):
   - Kubernetes deployment for scaling
   - Load balancing across multiple instances
   - Distributed caching with Redis

## Conclusion

The Heart Disease Prediction system architecture provides a robust, maintainable, and high-performance foundation for clinical prediction tasks. Its layered design with clear separation of concerns allows for easy extension and modification, while the performance optimizations ensure efficient operation even with large workloads.

The combination of multiple model types, ensemble methods, and sophisticated caching and parallel processing makes the system suitable for production deployment in clinical decision support scenarios.
