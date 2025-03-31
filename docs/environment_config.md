# Environment-Specific Configuration

This document describes how to configure and use the environment-specific settings in the Heart Disease Prediction system.

## Table of Contents

- [Overview](#overview)
- [Environment Types](#environment-types)
- [Configuration Files](#configuration-files)
- [Environment Variables](#environment-variables)
- [Running in Different Environments](#running-in-different-environments)
- [Docker Deployment](#docker-deployment)
- [Configuration Parameters](#configuration-parameters)
- [Best Practices](#best-practices)

## Overview

The Heart Disease Prediction system supports multiple deployment environments (development, staging, and production) with different configuration settings for each. This allows for:

- Different security settings between environments
- Optimized performance parameters for production
- More verbose logging in development
- Environment-specific API keys and authentication
- Different backup strategies per environment

## Environment Types

The system supports three standard environments:

1. **Development (`dev`)**:
   - Used for local development
   - Features hot-reloading, debug logging, and disabled security features
   - Optimized for quick iteration, not performance

2. **Staging (`staging`)**:
   - Production-like environment for testing
   - Security features enabled
   - Similar performance characteristics to production
   - Uses test data and credentials

3. **Production (`prod`)**:
   - Live environment for real-world use
   - Full security features enabled
   - Optimized for performance and reliability
   - Uses secure credentials and real data

## Configuration Files

Each environment has its own configuration file:

- `config/config.dev.yaml`: Development configuration
- `config/config.staging.yaml`: Staging configuration
- `config/config.prod.yaml`: Production configuration

The system automatically selects the appropriate configuration file based on the current environment.

## Environment Variables

The environment is determined by the `ENVIRONMENT` environment variable, which can be set to:

- `dev` or `development` for development
- `staging` or `stage` for staging
- `prod` or `production` for production

If not specified, the system defaults to `dev`.

Additionally, the configuration files can reference other environment variables using the `${VARIABLE_NAME}` syntax. This is particularly useful for sensitive information like API keys and database credentials.

## Running in Different Environments

### Using the run_environment.sh Script

The simplest way to run the system in a specific environment is using the `run_environment.sh` script:

```bash
./scripts/run_environment.sh --env=dev    # Development
./scripts/run_environment.sh --env=staging # Staging
./scripts/run_environment.sh --env=prod    # Production
```

This script:
1. Sets up the environment
2. Generates the appropriate .env file if it doesn't exist
3. Loads environment variables
4. Starts the system using Docker Compose with the environment-specific configuration

### Using the run_api.sh Script

For running just the API without Docker:

```bash
./scripts/run_api.sh --env=dev    # Development
./scripts/run_api.sh --env=staging # Staging
./scripts/run_api.sh --env=prod    # Production
```

### Direct Environment Variable Setting

You can also set the environment variable directly:

```bash
# Linux/macOS
export ENVIRONMENT=staging
python run_api.py

# Windows
set ENVIRONMENT=staging
python run_api.py
```

## Docker Deployment

For Docker deployment, there are environment-specific Docker Compose files:

- `docker-compose.dev.yaml`: Development configuration
- `docker-compose.staging.yaml`: Staging configuration
- `docker-compose.prod.yaml`: Production configuration

To use a specific environment:

```bash
docker-compose -f docker-compose.dev.yaml up -d     # Development
docker-compose -f docker-compose.staging.yaml up -d # Staging
docker-compose -f docker-compose.prod.yaml up -d    # Production
```

## Configuration Parameters

The configuration files contain environment-specific settings for various aspects of the system. Here are the key parameters that differ between environments:

### API Settings

| Parameter | Development | Staging | Production | Description |
|-----------|-------------|---------|------------|-------------|
| `log_level` | debug | info | warning | Logging verbosity |
| `batch_size` | 10 | 50 | 100 | Number of patients to process in each batch |
| `max_workers` | 2 | 4 | 8 | Number of parallel workers for batch processing |

### Authentication

| Parameter | Development | Staging | Production | Description |
|-----------|-------------|---------|------------|-------------|
| `enabled` | false | true | true | Whether authentication is required |
| `secret_key` | dev key | staging key | env variable | Secret key for JWT tokens |
| `access_token_expire_minutes` | 60 | 30 | 30 | Token expiration time in minutes |

### Caching

| Parameter | Development | Staging | Production | Description |
|-----------|-------------|---------|------------|-------------|
| `max_size` | 100 | 1000 | 5000 | Maximum number of cached predictions |
| `ttl` | 300 | 3600 | 7200 | Cache time-to-live in seconds |

### Backup

| Parameter | Development | Staging | Production | Description |
|-----------|-------------|---------|------------|-------------|
| `backup_frequency` | daily | daily | hourly | How often to run scheduled backups |
| `keep_backups` | 5 | 10 | 24 | Number of backups to keep |
| `cloud_backup` | false | true | true | Whether to upload backups to cloud storage |

### Environment-Specific Settings

| Parameter | Development | Staging | Production | Description |
|-----------|-------------|---------|------------|-------------|
| `debug` | true | false | false | Whether debug mode is enabled |
| `reload` | true | false | false | Whether to automatically reload on code changes |
| `testing` | false | true | false | Whether testing features are enabled |

## Best Practices

1. **Sensitive Information**: Never store sensitive information (passwords, API keys, etc.) in configuration files. Instead, use environment variables and reference them in the configuration.

2. **Production Deployment**: For production deployment, always use the production configuration and set appropriate environment variables for sensitive information.

3. **Testing Configuration**: Before deploying to production, test with the staging configuration to ensure everything works as expected.

4. **Environment Variables**:
   - Development: Can use default values from .env.dev
   - Staging: Should use environment-specific values
   - Production: Must use secure values set in the deployment environment

5. **Local Development**: Use the development environment for local development, but occasionally test with staging to catch environment-specific issues early.

6. **Configuration Changes**: When making changes to one environment's configuration, consider whether the changes should be applied to other environments as well.

7. **Documentation**: Update this document when adding new environment-specific configuration parameters.

## Troubleshooting

If you encounter issues with environment-specific configuration:

1. **Check Current Environment**: Use the API's `/health` endpoint to verify the current environment.

2. **Verify Configuration**: Check that the environment-specific configuration file exists and contains the expected settings.

3. **Environment Variables**: Ensure all required environment variables are set correctly.

4. **Logs**: Check the logs for warning or error messages related to configuration loading.

5. **Config Loading**: If the system is falling back to the default configuration, check that the environment variable is set correctly and the configuration file exists.
