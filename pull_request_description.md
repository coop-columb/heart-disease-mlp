# Codebase Cleanup and Testing Improvements

## Summary
This PR addresses various issues in the codebase, making it more robust and maintainable:
- Fixed Python path conflicts that were causing import errors with other projects
- Enhanced test reliability with improved error handling and skip conditions
- Made Docker tests more resilient with better error handling
- Consolidated duplicate configuration files
- Fixed API endpoints to properly handle model parameters
- Added better error handling for prediction endpoints

## Details

### Path and Import Fixes
- Fixed Python path handling in tests and API code to prevent conflicts with other projects
- Added explicit filtering of external project paths to ensure clean imports
- Made imports more resilient by adding try/except blocks where appropriate

### API Enhancements
- Added support for model specification in prediction endpoints
- Added batch prediction endpoint with proper error handling
- Made the API more resilient against missing models
- Added support for both Pydantic v1 and v2 serialization methods
- Improved risk level determination with better null handling

### Test Improvements
- Made Docker tests conditional on Docker availability
- Enhanced test skipping logic for better CI/CD compatibility
- Added better error reporting in tests
- Made model performance tests compatible with synthetic test data
- Fixed split_data test to handle small test datasets
- Made integration tests more robust

### Configuration
- Removed duplicate docker-compose.yml file in favor of more feature-rich yaml version
- Simplified deployment configuration

## Test plan
1. Run all tests with pytest to verify they pass
2. Verify API works with existing models through direct calls
3. Verify deployment scripts work correctly
4. Verify docker container builds and runs correctly
