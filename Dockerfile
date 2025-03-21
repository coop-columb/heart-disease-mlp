# Use an official Python runtime as a base image
FROM python:3.10-slim

# Set working directory in the container
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/models \
    PORT=8000

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

# Expose port
EXPOSE $PORT

# Set default command to run the API
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
