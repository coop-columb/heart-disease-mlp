# Simplified Deployment Guide

This guide provides straightforward instructions for deploying your Heart Disease Prediction API without relying on complex CI/CD workflows.

## Option 1: Direct Render Deployment (Simplest)

1. **Sign in to Render**: https://dashboard.render.com

2. **Create a New Web Service**:
   - Click "New +" button
   - Select "Web Service"
   - Connect your GitHub repo or choose "Deploy an existing image from a registry"

3. **Configure Your Service**:
   - Name: heart-disease-prediction-api
   - Environment: Docker
   - Branch: main (if using GitHub)
   - Region: (choose closest to you)
   - Plan: Free

4. **Set Environment Variables**:
   - PORT: 8000
   - MODEL_DIR: /app/models

5. **Create Web Service**

6. **Monitor Build Progress**:
   - Wait for the build and deployment to complete (usually 5-10 minutes)
   - Check logs for any issues

7. **Access Your API**:
   - Use the provided URL (e.g., https://heart-disease-prediction-api.onrender.com)
   - Test with: `curl https://heart-disease-prediction-api.onrender.com/health`

## Option 2: Local Docker Deployment

If you prefer to run locally or on your own server:

1. **Build the Docker image**:
   ```bash
   docker build -t heart-disease-mlp .
   ```

2. **Run the container**:
   ```bash
   docker run -p 8000:8000 heart-disease-mlp
   ```

3. **Access your API**:
   - Open: http://localhost:8000/health
   - API documentation: http://localhost:8000/docs

## Option 3: Deploy to DigitalOcean App Platform

DigitalOcean has a very simple deployment process:

1. **Create a DigitalOcean account**: https://digitalocean.com

2. **Create a new App**:
   - Select GitHub repository
   - Choose Dockerfile as source
   - Select a region and plan (Basic/Starter is sufficient)
   - Set environment variables PORT=8000 and MODEL_DIR=/app/models
   - Deploy

## Option 4: Deploy to Fly.io

Fly.io offers simple deployment:

1. **Install flyctl CLI**:
   ```bash
   curl -L https://fly.io/install.sh | sh
   ```

2. **Login and deploy**:
   ```bash
   fly auth login
   fly launch
   ```

3. **Follow the interactive prompts** to configure your app

## Testing Your Deployment

Regardless of where you deploy, you can test your API with:

1. **Health check**:
   ```
   curl https://your-deployment-url.com/health
   ```

2. **Sample prediction**:
   ```
   curl -X POST https://your-deployment-url.com/predict \
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

## Simple Production Setup

For a simple but robust production setup, consider:

1. **Docker + Nginx**:
   - Deploy to any VPS (DigitalOcean, Linode, AWS EC2)
   - Use Nginx as a reverse proxy
   - Set up SSL with Let's Encrypt

2. **Docker Compose**:
   ```yaml
   version: '3'
   services:
     api:
       build: .
       ports:
         - "8000:8000"
       environment:
         - PORT=8000
         - MODEL_DIR=/app/models
   ```

This approach gives you full control without the complexity of CI/CD pipelines.