# Railway Deployment Troubleshooting

This guide provides solutions to common issues encountered when deploying the Heart Disease Prediction System to Railway using GitHub Actions.

## Common Issues and Solutions

### 1. Authentication Failures

**Symptoms:**
- "Not authenticated" errors
- "Failed to login" errors
- The deployment workflow fails at the `railway login` step

**Solutions:**
- Verify your `RAILWAY_TOKEN` secret is correctly set in GitHub repository settings
- Generate a new Railway token if the current one might be expired
- Ensure the token has the necessary permissions (usually requires full access)
- Try the explicit browserless login: `railway login --browserless`

### 2. Project Linking Issues

**Symptoms:**
- "No project linked" errors
- "Failed to link project" errors

**Solutions:**
- If you have multiple Railway projects, specify the project ID:
  ```bash
  railway link --project YOUR_PROJECT_ID
  ```
- Create a new project if linking fails:
  ```bash
  railway init --name heart-disease-prediction
  ```
- Check if the project already exists in your Railway dashboard

### 3. Deployment Failures

**Symptoms:**
- Build fails after successful login and project linking
- "Failed to deploy" errors

**Solutions:**
- Check that your `railway.json` configuration is valid
- Ensure your Dockerfile is properly configured
- Verify all necessary files are included in your repository
- Try running with verbose logging: `railway up --verbose`
- Check Railway dashboard for build logs and error messages

### 4. Resource Limitations

**Symptoms:**
- Deployment succeeds but application crashes
- Out of memory errors in logs

**Solutions:**
- Check if you're exceeding Railway's free tier limits
- Optimize your Docker image size and resource usage
- Consider upgrading to a paid plan if necessary

### 5. Environment Variable Issues

**Symptoms:**
- Application deploys but doesn't function correctly
- Configuration errors in logs

**Solutions:**
- Set necessary environment variables in Railway dashboard
- Check that your app correctly reads environment variables
- Verify that sensitive information is not hardcoded

### 6. Action Workflow Debug Process

If the GitHub Action workflow fails, follow these debugging steps:

1. **Check workflow logs:**
   - Go to your repository on GitHub
   - Click "Actions" tab
   - Find the failed workflow run
   - Examine the detailed logs for each step

2. **Test locally:**
   ```bash
   # Login to Railway
   railway login

   # Link to project
   railway link

   # Try deploying
   railway up
   ```

3. **Compare configurations:**
   - Verify your local setup matches the GitHub Actions workflow
   - Check that your `railway.json` file is correct

## Getting Additional Help

If you're still experiencing issues:

1. Check the [Railway documentation](https://docs.railway.app/)
2. Visit the [Railway Discord community](https://discord.railway.app/)
3. Create an issue in this repository with:
   - A description of the problem
   - Relevant logs (with sensitive information removed)
   - Steps you've already tried

## Common Railway CLI Commands

```bash
# Login to Railway
railway login

# Link to a project
railway link --project YOUR_PROJECT_ID

# Initialize a new project
railway init --name PROJECT_NAME

# Deploy current directory
railway up

# View project status
railway status

# View logs
railway logs

# List all projects
railway projects

# List all environments
railway environments
```
