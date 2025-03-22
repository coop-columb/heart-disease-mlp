# Deployment Secrets Checklist

Use this file as a reminder of the secrets you need to configure in GitHub.
Check off each item as you complete it.

## Railway Deployment (Recommended)

Add the following secret for Railway deployment:

- [ ] `RAILWAY_TOKEN`: Your Railway API token (Get from Railway dashboard → Settings → API Tokens)

## Traditional Server Deployment (Alternative)

If you prefer to use traditional server deployment instead of Railway, add these secrets:

- [x] `SSH_PRIVATE_KEY`: The SSH private key you generated (already added)
- [ ] `DEPLOY_HOST`: Hostname/IP for staging server
- [ ] `DEPLOY_USER`: Username for staging server
- [ ] `PROD_DEPLOY_HOST`: Hostname/IP for production server
- [ ] `PROD_DEPLOY_USER`: Username for production server

## Steps for Setting Up Servers

1. Add your public key to the authorized_keys file on your servers:

```bash
# Replace with your actual server details
ssh-copy-id -i ~/.ssh/heart-disease-deploy/deploy_key.pub user@staging-server
ssh-copy-id -i ~/.ssh/heart-disease-deploy/deploy_key.pub user@production-server
```

2. Create the project directory on both servers:

```bash
ssh user@staging-server "mkdir -p ~/heart-disease-mlp"
ssh user@production-server "mkdir -p ~/heart-disease-mlp"
```

3. Make sure Docker and Docker Compose are installed on both servers

## Testing Deployment

After adding all secrets to GitHub:

1. Make a minor change to any file
2. Commit and push to the main branch
3. Go to the GitHub Actions tab to monitor the workflow execution

## Your Generated SSH Key

You've successfully generated:
- Public key: ~/.ssh/heart-disease-deploy/deploy_key.pub
- Private key: ~/.ssh/heart-disease-deploy/deploy_key

The public key should be added to your servers, and the private key content should be added as the `SSH_PRIVATE_KEY` secret in GitHub.
