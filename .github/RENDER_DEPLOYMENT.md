# Render Deployment Guide

This guide will help you deploy your Heart Disease Prediction System on Render.com, a simple cloud platform ideal for applications like this one.

## Why Render?

- Simple setup process
- Free tier for hobby projects
- Support for Docker-based deployments
- Automatic deployments from GitHub
- Health checks for reliability
- Custom domains support

## Setup Instructions

### Option 1: Manual Deployment (Recommended First Time)

1. Create a Render account at https://render.com/
2. Connect your GitHub account in the Render dashboard
3. Click "New +" and select "Web Service"
4. Select your heart-disease-mlp repository
5. Configure with these settings:
   - Name: heart-disease-prediction-api
   - Environment: Docker
   - Branch: main
   - Region: (choose closest to you)
   - Plan: Free

6. Create Web Service

Render will automatically:
- Detect the Dockerfile
- Build the Docker image
- Deploy the application
- Give you a public URL

### Option 2: Automatic Deployment via GitHub Actions

For fully automated CI/CD, we've included a GitHub Actions workflow:

1. Get your Render API key:
   - Go to Render dashboard → Account Settings → API Keys
   - Create a new API key with an appropriate description

2. Get your service ID:
   - Go to your service in the Render dashboard
   - The service ID is in the URL: `https://dashboard.render.com/web/srv-xxxxxxxxxxxx`
   - The ID is the `srv-xxxxxxxxxxxx` part

3. Add these as GitHub secrets:
   - Name: `RENDER_API_KEY`
   - Value: Your Render API key
   - Name: `RENDER_SERVICE_ID`
   - Value: Your service ID (e.g. `srv-xxxxxxxxxxxx`)
   - Name: `RENDER_SERVICE_URL` (optional)
   - Value: Your service URL (e.g. `https://heart-disease-prediction-api.onrender.com`)

The workflow will:
- Run basic tests
- Deploy to Render
- Verify deployment health

## Checking Deployment Status

After deployment:
1. Go to your Render dashboard
2. Click on your heart-disease-prediction service
3. Check the "Logs" tab for any issues
4. Get your service URL from the "Overview" tab

## Testing Your Deployment

To verify your deployment is working:

1. Use the health endpoint:
   ```
   curl https://your-render-url.onrender.com/health
   ```

2. Test a prediction:
   ```
   curl -X POST https://your-render-url.onrender.com/predict \
     -H "Content-Type: application/json" \
     -d '{
       "age": 61,
       "sex": 1,
       "cp": 3,
       "trestbps": 140,
       "chol": 240,
       "fbs": 1,
       "restecg": 1,
       "thalach": 150,
       "exang": 1,
       "oldpeak": 2.4,
       "slope": 2,
       "ca": 1,
       "thal": 3
     }'
   ```

## Advanced Configuration

### Custom Domain (Optional)

To use a custom domain:
1. Go to your service in Render dashboard
2. Click "Settings" and scroll to "Custom Domain"
3. Follow the instructions to configure your DNS

### Environment Variables (Optional)

To add environment variables:
1. Go to your service in Render dashboard
2. Click "Environment" tab
3. Add key-value pairs as needed

## Troubleshooting

Common issues and solutions:

1. **Build failures**: 
   - Check your Dockerfile for errors
   - Ensure all dependencies are correctly specified

2. **Service crashes**:
   - Check the logs in the Render dashboard
   - Ensure the health check endpoint is working

3. **Memory issues**:
   - Consider upgrading from the free plan if your model requires more memory

## Need Help?

If you encounter issues with Render deployment:
1. Check [Render documentation](https://render.com/docs)
2. Visit the [Render support page](https://render.com/docs/support)
3. Create an issue in this repository with:
   - A description of the problem
   - Relevant logs (with sensitive information removed)
   - Steps you've already tried