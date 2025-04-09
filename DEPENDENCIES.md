# Project Dependencies Investigation Wed Apr  9 18:55:51 EDT 2025

## Dependencies Found During Testing

### Core ML & API
- numpy, pandas, scikit-learn, tensorflow
- fastapi, uvicorn[standard]
- matplotlib, seaborn (visualization)
- python-jose[cryptography], PyJWT
- python-multipart, passlib[bcrypt]

### Development
- pytest, pytest-cov
- black, isort, flake8
- httpx (FastAPI testing)

## Issues Found
1. Missing test dependencies discovered during testing
2. Pydantic v2 deprecation warnings
3. ML-specific warnings in test suite

## Next Steps
1. Update pyproject.toml
2. Address warnings
3. Improve test configuration
