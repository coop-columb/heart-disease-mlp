## Summary
This PR implements a complete model deployment solution for the heart disease prediction system:

- Created prediction module for loading trained models and making predictions
- Implemented a command-line interface for easy predictions from files
- Built a FastAPI-based REST API for model serving
- Added Docker and docker-compose configuration for containerization
- Added comprehensive README with setup and deployment instructions

## Test plan
1. Verify the prediction module works with existing trained models
2. Test the CLI with the sample patient data
3. Start the FastAPI server and test the /predict endpoint
4. Build and run the Docker container to ensure it starts properly
5. Verify API documentation is available at /docs
