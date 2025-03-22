# CI/CD Pipeline Documentation

This directory contains GitHub Actions workflow files that automate testing, building, and deployment of the Heart Disease Prediction system.

## Workflows

### 1. Main CI/CD Pipeline (`main.yml`)

The main pipeline runs on push to main branch, pull requests to main, and can be manually triggered. It consists of four sequential stages:

- **Lint and Test**: Runs on multiple Python versions (3.8, 3.9, 3.10)
  - Code linting with flake8
  - Code formatting checks with black and isort
  - Unit and integration tests with pytest and coverage reporting

- **Build and Push**: Builds a Docker image and pushes it to GitHub Container Registry
  - Only runs on pushes to the main branch
  - Tags images with both 'latest' and the commit SHA

- **Deploy to Staging**: Deploys the newly built image to a staging environment
  - Uses SSH to connect to the staging server
  - Updates the Docker image and restarts the application
  - Requires configured secrets: `SSH_PRIVATE_KEY`, `DEPLOY_HOST`, `DEPLOY_USER`

- **Deploy to Production**: Deploys to the production environment
  - Requires manual approval via GitHub environment protection rules
  - Updates the Docker image and restarts the application
  - Requires configured secrets: `SSH_PRIVATE_KEY`, `PROD_DEPLOY_HOST`, `PROD_DEPLOY_USER`

### 2. Model Retraining (`model-retraining.yml`)

Scheduled workflow that retrains the machine learning models:

- Runs monthly on the 1st at 3am, or can be manually triggered
- Downloads latest data, processes it, and retrains models
- Creates a pull request with the updated model files
- Includes evaluation metrics in the PR description

### 3. Security Scanning (`security-scan.yml`)

Security scanning workflow for continuous security monitoring:

- Runs on pushes to main, PRs to main, weekly on Mondays, or manual triggers
- Scans dependencies for vulnerabilities using Safety
- Performs static code analysis with Bandit
- Scans Docker images with Trivy
- Uploads security reports as artifacts and SARIF reports

## Required Secrets

To use these workflows, the following GitHub secrets need to be configured:

- `GITHUB_TOKEN`: Automatically provided by GitHub Actions
- `SSH_PRIVATE_KEY`: SSH private key for deployment access
- `DEPLOY_HOST`: Hostname or IP of the staging server
- `DEPLOY_USER`: Username for SSH access to staging server
- `PROD_DEPLOY_HOST`: Hostname or IP of the production server
- `PROD_DEPLOY_USER`: Username for SSH access to production server

## Environment Configuration

The workflows use GitHub Environments for deployment management:

1. **staging**: For the staging deployment stage
   - Configure this environment in GitHub repository settings

2. **production**: For the production deployment stage
   - Configure with required protection rules (approval reviews)
   - Set this up in GitHub repository settings

## Local Testing

Workflows can be tested locally using [act](https://github.com/nektos/act), a tool for running GitHub Actions locally:

```bash
# Install act (requires Docker)
# macOS: brew install act

# Run the entire main workflow
act -j lint-and-test

# Run a specific job
act -j security-scan
```
