# Railway Deployment Guide

This guide will help you deploy your Heart Disease Prediction System on Railway.

## Pre-requisites

You mentioned you already have a Railway account, which is great! If you haven't already, make sure to:

1. Connect your GitHub account to Railway
2. Install the Railway CLI (optional but useful)
   ```bash
   npm i -g @railway/cli
   ```

## Deploying to Railway

### Option 1: Deploy via Railway Dashboard (Easiest)

1. Log in to your Railway account: https://railway.app/dashboard
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Choose your heart-disease-mlp repository
5. Railway will automatically detect your Dockerfile and use it
6. Configure your settings:
   - Environment: Production
   - Root Directory: / (root)
   - Service Name: heart-disease-prediction-api

7. Click "Deploy"

Railway will now:
- Pull your code
- Build the Docker image
- Deploy the application
- Provide you with a public URL

### Option 2: Using GitHub Actions to Deploy to Railway

You can also modify your GitHub Actions workflow to deploy to Railway:

1. Get a Railway API key:
   - Go to Railway dashboard -> Settings -> API Tokens
   - Generate a new token with appropriate permissions

2. Add the token as a GitHub secret:
   - Name: `RAILWAY_TOKEN`
   - Value: Your Railway API token

3. Add a new file `.github/workflows/railway-deploy.yml`:

```yaml
name: Deploy to Railway

on:
  push:
    branches: [ main ]
  workflow_dispatch:

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v3

      - name: Install Railway CLI
        run: npm i -g @railway/cli

      - name: Deploy to Railway
        run: railway up
        env:
          RAILWAY_TOKEN: ${{ secrets.RAILWAY_TOKEN }}
```

## Monitoring Your Deployment

After deployment:
1. Go to your Railway dashboard
2. Click on your heart-disease-prediction project
3. You'll see metrics, logs, and the deployment URL

## Continuous Deployment

Railway will automatically redeploy your application whenever you push to your main branch.

## Custom Domain (Optional)

To use a custom domain:
1. Go to your Railway project settings
2. Click on "Domains"
3. Add your custom domain
4. Follow the instructions to set up DNS records

## Environment Variables (Optional)

If your application needs environment variables:
1. Go to your Railway project
2. Click on "Variables"
3. Add any required variables

## Limitations

Railway's free tier includes:
- 5 projects
- Shared CPU
- 512 MB RAM
- 1 GB disk
- 5 minute execution timeout

Consider upgrading to a paid plan for production workloads.

## Need Help?

If you encounter any issues, check Railway's documentation:
https://docs.railway.app/
