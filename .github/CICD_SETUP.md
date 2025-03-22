# CI/CD Setup Guide

This guide explains how to set up the CI/CD pipeline for the Heart Disease MLP project.

## Required GitHub Secrets

To enable the full functionality of the CI/CD pipeline, you need to configure the following secrets in your GitHub repository:

1. Navigate to your repository on GitHub
2. Go to Settings > Secrets and variables > Actions
3. Add the following repository secrets:

### Deployment Secrets

| Secret Name | Description |
|-------------|-------------|
| `SSH_PRIVATE_KEY` | SSH private key for deployment access (without passphrase) |
| `DEPLOY_HOST` | Hostname or IP of the staging server |
| `DEPLOY_USER` | Username for SSH access to staging server |
| `PROD_DEPLOY_HOST` | Hostname or IP of the production server |
| `PROD_DEPLOY_USER` | Username for SSH access to production server |

## GitHub Environments

The pipeline uses GitHub Environments for deployment management and approval workflows:

1. Go to your repository on GitHub
2. Navigate to Settings > Environments
3. Create two environments:
   - `staging`
   - `production`

### Production Environment Protection Rules

For the production environment, add protection rules:

1. Required reviewers: Add team members who should approve production deployments
2. Wait timer: Consider adding a wait period (e.g., 10 minutes) before production deployments

## Container Registry Access

To enable pushing Docker images to GitHub Container Registry:

1. Go to your repository on GitHub
2. Navigate to Settings > Actions > General
3. Under "Workflow permissions", select:
   - "Read and write permissions"
   - Check "Allow GitHub Actions to create and approve pull requests"

## Server Setup

Ensure the following on your deployment servers:

1. Docker and Docker Compose are installed
2. The deployment user has permissions to run Docker commands
3. The SSH key specified in `SSH_PRIVATE_KEY` is authorized on the server
4. Project directory structure is properly set up:
   ```
   ~/heart-disease-mlp/
   └── docker-compose.yml
   ```

## Monitoring Setup

After configuring the CI/CD pipeline, monitor its performance:

1. Check GitHub Actions runs to ensure all jobs complete successfully
2. Verify deployments reach staging and production environments
3. Consider setting up notifications for workflow failures:
   - Go to your GitHub profile > Settings > Notifications
   - Configure preferences for workflow run notifications

## Troubleshooting

- **SSH Connection Issues**: Ensure the SSH key is correctly formatted and doesn't have a passphrase
- **Docker Build Failures**: Check for Docker Hub rate limits; consider authenticating with Docker Hub
- **Deployment Failures**: Verify server disk space, Docker daemon status, and network accessibility
