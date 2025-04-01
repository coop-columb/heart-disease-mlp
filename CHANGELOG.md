# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-04-01

Initial release with core functionality for heart disease prediction API.

### Added
- FastAPI-based REST API for heart disease prediction
- Two ML models integrated:
  - Scikit-learn MLP Classifier
  - Keras/TensorFlow Deep Neural Network
- Authentication system with:
  - JWT token-based authentication
  - API key authentication
  - Public endpoints configuration
- Configuration management system:
  - Environment-specific configurations (dev, staging, prod)
  - Environment variable resolution
- Comprehensive test suite:
  - Authentication tests
  - API endpoint tests
  - Model prediction tests
- Model preprocessing pipeline:
  - Feature scaling
  - Categorical encoding
  - Data validation
- API features:
  - Batch prediction support
  - Performance logging
  - Request caching
  - Configurable worker pool

### Technical Features
- Modular project architecture
- Environment-based configuration system
- Automated testing setup with pytest
- Type hints and docstrings for better code documentation
- Error handling and validation
- Logging system for debugging and monitoring

[0.1.0]: https://github.com/coop-columb/heart-disease-mlp/releases/tag/v0.1.0
