# CI/CD Setup Guide

This document provides instructions for setting up the CI/CD pipeline for the Heart Disease Prediction System.

## GitHub Environments Setup

The deployment workflow uses GitHub Environments for staging and production deployments. To set these up:

1. Go to your GitHub repository
2. Click on "Settings" > "Environments"
3. Click "New environment"
4. Create two environments:
   - `staging`
   - `production`
5. For each environment, you can set up:
   - Environment protection rules (e.g., required reviewers)
   - Environment secrets (see below)

## Required Secrets

The following secrets need to be added to your GitHub repository:

### Repository Secrets

1. `SSH_PRIVATE_KEY`: The SSH private key for connecting to deployment servers

### Environment Secrets (for staging)

1. `DEPLOY_HOST`: The hostname/IP of your staging server
2. `DEPLOY_USER`: The username for SSH access to your staging server

### Environment Secrets (for production)

1. `PROD_DEPLOY_HOST`: The hostname/IP of your production server
2. `PROD_DEPLOY_USER`: The username for SSH access to your production server

## Workflow Configuration

After setting up environments and secrets, uncomment the `environment` lines in `.github/workflows/main.yml`:

```yml
deploy-staging:
  # ...
  environment: staging  # Uncomment this line
  # ...

deploy-production:
  # ...
  environment: production  # Uncomment this line
  # ...
```

## Manual Pipeline Triggering

You can manually trigger the CI/CD pipeline:

1. Go to your GitHub repository
2. Click on "Actions"
3. Select "CI/CD Pipeline" from the workflows list
4. Click "Run workflow"
5. Select the branch to run it on
6. Click "Run workflow"

## Model Retraining

You can manually trigger model retraining:

1. Go to your GitHub repository
2. Click on "Actions"
3. Select "Model Retraining and Evaluation" from the workflows list
4. Click "Run workflow"
5. Configure retraining options:
   - Enable/disable hyperparameter tuning
   - Set number of tuning trials (if tuning is enabled)
6. Click "Run workflow"

The retraining workflow is also scheduled to run automatically on the 1st of each month at 3am UTC.

## Security Scanning

The security scanning workflow runs automatically:
- On pushes to main
- On pull requests to main
- Weekly (every Monday at 1am UTC)

You can also trigger it manually through the Actions tab.
