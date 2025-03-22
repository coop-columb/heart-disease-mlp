#!/usr/bin/env python
"""
Deployment debugging script for Heart Disease Prediction API.
This script checks for common issues that might prevent successful deployment.
"""

import os
import sys
import json
import subprocess
from pathlib import Path

def print_section(title):
    """Print a section title."""
    print("\n" + "=" * 50)
    print(f" {title} ".center(50, "="))
    print("=" * 50 + "\n")

def check_file_exists(filepath, required=True):
    """Check if a file exists and print its status."""
    path = Path(filepath)
    status = "✅ Present" if path.exists() else "❌ Missing"
    if not path.exists() and required:
        status += " (REQUIRED)"
    print(f"{filepath}: {status}")
    return path.exists()

def read_file_content(filepath):
    """Read and return file content, or None if file doesn't exist."""
    path = Path(filepath)
    if not path.exists():
        return None
    try:
        with open(path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

def main():
    """Run deployment debugging checks."""
    # Get project root directory
    project_root = Path(__file__).parent.parent.absolute()
    os.chdir(project_root)
    
    print_section("DEPLOYMENT DEBUGGING")
    print(f"Project root: {project_root}")
    
    # Check critical deployment files
    print_section("CRITICAL FILES")
    dockerfile_exists = check_file_exists("Dockerfile", required=True)
    check_file_exists("entrypoint.sh", required=True)
    check_file_exists("requirements.txt", required=True)
    check_file_exists("render.yaml", required=False)
    check_file_exists("api/app.py", required=True)
    
    # Check Docker configuration
    if dockerfile_exists:
        print_section("DOCKERFILE ANALYSIS")
        dockerfile = read_file_content("Dockerfile")
        print(dockerfile)
        
        # Basic Dockerfile checks
        if "ENTRYPOINT" not in dockerfile and "CMD" not in dockerfile:
            print("\n❌ ERROR: Dockerfile missing ENTRYPOINT or CMD directive")
        if "EXPOSE" not in dockerfile:
            print("\n⚠️ WARNING: Dockerfile missing EXPOSE directive")
        if "ENV PORT" not in dockerfile:
            print("\n⚠️ WARNING: PORT environment variable not set in Dockerfile")
    
    # Check entrypoint script
    print_section("ENTRYPOINT SCRIPT")
    entrypoint = read_file_content("entrypoint.sh")
    if entrypoint:
        print(entrypoint)
        
        # Check for executable permission
        try:
            file_stats = os.stat("entrypoint.sh")
            is_executable = bool(file_stats.st_mode & 0o111)
            status = "✅ Executable" if is_executable else "❌ NOT executable"
            print(f"\nentrypoint.sh permissions: {status}")
        except Exception as e:
            print(f"Error checking permissions: {str(e)}")

    # Check API implementation
    print_section("API IMPLEMENTATION")
    app_py = read_file_content("api/app.py")
    if app_py:
        print("API app.py found, checking for crucial components...")
        if "/health" in app_py:
            print("✅ Health endpoint defined")
        else:
            print("❌ ERROR: No /health endpoint found in app.py")
        
        if "app = FastAPI" in app_py:
            print("✅ FastAPI app initialized")
        else:
            print("❌ ERROR: FastAPI app not initialized correctly")
    
    # Check model loading
    print_section("MODEL LOADING")
    if "HeartDiseasePredictor" in app_py:
        print("✅ Model predictor class referenced in app.py")
    else:
        print("❌ ERROR: Model predictor not properly referenced")
    
    # Directory structure
    print_section("DIRECTORY STRUCTURE")
    for dir_name in ["api", "data", "models", "src"]:
        dir_exists = os.path.isdir(dir_name)
        status = "✅ Present" if dir_exists else "❌ Missing"
        print(f"{dir_name}/: {status}")
    
    # Render.yaml analysis if it exists
    if os.path.exists("render.yaml"):
        print_section("RENDER.YAML ANALYSIS")
        try:
            with open("render.yaml", 'r') as f:
                render_config = f.read()
                print(render_config)
        except Exception as e:
            print(f"Error reading render.yaml: {str(e)}")

    # Provide deployment recommendations
    print_section("DEPLOYMENT RECOMMENDATIONS")
    print("Based on the analysis, here are the recommendations:")
    
    if not dockerfile_exists:
        print("1. Create a Dockerfile to define how to build your application")
    
    print("""
1. When deploying to Render, ensure these environment variables are set:
   - PORT=8000
   - MODEL_DIR=/app/models

2. Make sure the entrypoint.sh script is executable:
   chmod +x entrypoint.sh

3. Verify your models directory structure:
   - /app/models/ should contain your trained models

4. For Render deployment issues:
   - Check build logs in the Render dashboard
   - Verify the health endpoint is working
   - Check CPU/memory usage in Render dashboard

5. For Docker-specific problems:
   - Test the Docker image locally before deploying
   - Ensure the container exposes port 8000
   - Verify app properly reads environment variables
""")

if __name__ == "__main__":
    main()