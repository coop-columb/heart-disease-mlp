# Deployment Secrets Guide

This guide explains how to set up the secrets required for the CI/CD deployment workflow.

## Required Secrets

The workflow needs the following secrets:

1. `SSH_PRIVATE_KEY`: An SSH key for secure connection to your deployment servers
2. `DEPLOY_HOST`: The hostname or IP address of your staging server
3. `DEPLOY_USER`: The username to SSH into your staging server
4. `PROD_DEPLOY_HOST`: The hostname or IP address of your production server
5. `PROD_DEPLOY_USER`: The username to SSH into your production server

## Creating an SSH Key Pair

If you don't already have an SSH key for deployment:

1. Generate a new SSH key pair:
   ```bash
   ssh-keygen -t ed25519 -C "deployment-key" -f ~/.ssh/deployment_key
   ```

2. This creates two files:
   - `~/.ssh/deployment_key` (private key)
   - `~/.ssh/deployment_key.pub` (public key)

3. The private key will be used as the `SSH_PRIVATE_KEY` in GitHub secrets

4. Add the public key to the authorized_keys file on your deployment servers:
   ```bash
   ssh-copy-id -i ~/.ssh/deployment_key.pub user@staging-server
   ssh-copy-id -i ~/.ssh/deployment_key.pub user@production-server
   ```

## Adding Secrets to GitHub

### Repository Secrets

1. Go to your GitHub repository
2. Click on "Settings" > "Secrets and variables" > "Actions"
3. Click "New repository secret"
4. Add `SSH_PRIVATE_KEY`:
   - Name: `SSH_PRIVATE_KEY`
   - Value: The entire contents of your private key file (e.g., `~/.ssh/deployment_key`)
   - Make sure to include the full key including the begin/end lines

### Environment Secrets

You need to create environments first:
1. Go to "Settings" > "Environments"
2. Create "staging" environment
3. Add environment secrets:
   - `DEPLOY_HOST`: Your staging server hostname or IP (e.g., `staging.example.com` or `192.168.1.100`)
   - `DEPLOY_USER`: Username for SSH access (e.g., `deploy`)

4. Create "production" environment
5. Add environment secrets:
   - `PROD_DEPLOY_HOST`: Your production server hostname or IP (e.g., `production.example.com`)
   - `PROD_DEPLOY_USER`: Username for SSH access (e.g., `deploy`)

## Server Preparation

On both staging and production servers:

1. Create the project directory:
   ```bash
   mkdir -p ~/heart-disease-mlp
   ```

2. Install Docker and Docker Compose

3. Set up permissions for the deployment user to run Docker commands without sudo:
   ```bash
   sudo usermod -aG docker $USER
   ```

4. Create a basic docker-compose.yml file in the project directory:
   ```bash
   cd ~/heart-disease-mlp
   cat > docker-compose.yml <<EOL
   version: '3'
   services:
     app:
       image: ghcr.io/yourusername/heart-disease-mlp:latest
       ports:
         - "8000:8000"
       restart: unless-stopped
   EOL
   ```

5. Configure the server for GitHub Container Registry access if needed

## Uncomment Environment References

After setting up all secrets and environments, uncomment the environment lines in `.github/workflows/main.yml`:

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

## Testing the Deployment

1. Make a small change to the repository
2. Commit and push to the main branch
3. Go to the Actions tab and monitor the workflow execution
4. Check for successful deployment to the staging environment
5. After approval, it should deploy to production
