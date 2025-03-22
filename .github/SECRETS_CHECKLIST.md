# Deployment Secrets Checklist

Use this file as a reminder of the secrets you need to configure in GitHub.
Check off each item as you complete it.

## Repository Secrets

- [ ] `SSH_PRIVATE_KEY`: Your SSH private key for deployment

## Staging Environment Secrets

- [ ] Create "staging" environment in GitHub repository settings
- [ ] `DEPLOY_HOST`: Hostname/IP for staging server
- [ ] `DEPLOY_USER`: Username for staging server

## Production Environment Secrets

- [ ] Create "production" environment in GitHub repository settings
- [ ] `PROD_DEPLOY_HOST`: Hostname/IP for production server
- [ ] `PROD_DEPLOY_USER`: Username for production server

## After Configuration

- [ ] Uncomment environment references in .github/workflows/main.yml
- [ ] Test deployment by pushing a minor change to main branch

For detailed instructions, see [Deployment Secrets Guide](.github/DEPLOYMENT_SECRETS.md)
