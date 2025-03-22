# Use an official Python runtime as a base image
FROM python:3.10-slim

# Set working directory in the container
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/models \
    PORT=8000

# Print environment variables during build for debugging
RUN echo "Environment variables set: PORT=${PORT}, MODEL_DIR=${MODEL_DIR}"

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove gcc g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the project files into the container
COPY . .

# Make the entrypoint script executable
RUN chmod +x /app/entrypoint.sh

# Create necessary directories
RUN mkdir -p /app/data/processed /app/models

# Expose port
EXPOSE $PORT

# Set entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]
