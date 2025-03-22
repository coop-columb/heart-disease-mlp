# Deployment Secrets Guide

This guide explains how to set up the secrets required for the CI/CD deployment workflow.

## Required Secrets

The workflow needs the following secrets:

1. `SSH_PRIVATE_KEY`: An SSH key for secure connection to your deployment servers
2. `DEPLOY_HOST`: The hostname or IP address of your staging server
3. `DEPLOY_USER`: The username to SSH into your staging server
4. `PROD_DEPLOY_HOST`: The hostname or IP address of your production server
5. `PROD_DEPLOY_USER`: The username to SSH into your production server

## Your Generated SSH Key Pair

You've already generated an SSH key pair:

- Public key: `~/.ssh/heart-disease-deploy/deploy_key.pub`
- Private key: `~/.ssh/heart-disease-deploy/deploy_key`

The private key will be used as the `SSH_PRIVATE_KEY` in GitHub secrets.

To use this key for deployment, add the public key to the authorized_keys file on your deployment servers:

```bash
ssh-copy-id -i ~/.ssh/heart-disease-deploy/deploy_key.pub user@staging-server
ssh-copy-id -i ~/.ssh/heart-disease-deploy/deploy_key.pub user@production-server
```

## Adding Secrets to GitHub

### Repository Secrets

1. Go to your GitHub repository
2. Click on "Settings" > "Secrets and variables" > "Actions"
3. Click "New repository secret"
4. Add all required secrets:

   a. `SSH_PRIVATE_KEY`:
      - Name: `SSH_PRIVATE_KEY`
      - Value: The entire contents of your private key file (`~/.ssh/heart-disease-deploy/deploy_key`)
      - Make sure to include the full key including the begin/end lines

   b. `DEPLOY_HOST`:
      - Name: `DEPLOY_HOST`
      - Value: Your staging server hostname or IP (e.g., `staging.example.com` or `192.168.1.100`)

   c. `DEPLOY_USER`:
      - Name: `DEPLOY_USER`
      - Value: Username for SSH access to staging (e.g., `deploy`)

   d. `PROD_DEPLOY_HOST`:
      - Name: `PROD_DEPLOY_HOST`
      - Value: Your production server hostname or IP

   e. `PROD_DEPLOY_USER`:
      - Name: `PROD_DEPLOY_USER`
      - Value: Username for SSH access to production

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

## Testing the Deployment

1. Make a small change to the repository
2. Commit and push to the main branch
3. Go to the Actions tab and monitor the workflow execution
4. Check for successful deployment to the staging environment
5. After approval, it should deploy to production
