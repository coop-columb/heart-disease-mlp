"""
Test fixtures for deployment validation.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pytest
import yaml
from pydantic import BaseModel

class TestPatient(BaseModel):
    """Test patient data structure."""
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

class TestPrediction(BaseModel):
    """Expected prediction structure."""
    prediction: int
    probability: float
    risk_level: str
    interpretation: str
    model_used: str

@pytest.fixture
def environment_config():
    """Load environment configuration."""
    config_path = Path("config/environments.yml")
    with open(config_path) as f:
        return yaml.safe_load(f)

@pytest.fixture
def test_patients() -> List[Dict]:
    """Generate test patient data."""
    return [
        # Known negative case
        {
            "age": 45,
            "sex": 0,
            "cp": 0,
            "trestbps": 120,
            "chol": 180,
            "fbs": 0,
            "restecg": 0,
            "thalach": 170,
            "exang": 0,
            "oldpeak": 0.2,
            "slope": 1,
            "ca": 0,
            "thal": 2
        },
        # Known positive case
        {
            "age": 65,
            "sex": 1,
            "cp": 2,
            "trestbps": 150,
            "chol": 280,
            "fbs": 1,
            "restecg": 2,
            "thalach": 130,
            "exang": 1,
            "oldpeak": 2.5,
            "slope": 2,
            "ca": 2,
            "thal": 3
        }
    ]

@pytest.fixture
def expected_predictions() -> List[TestPrediction]:
    """Expected prediction results."""
    return [
        TestPrediction(
            prediction=0,
            probability=0.15,
            risk_level="LOW",
            interpretation="Low risk based on normal vitals",
            model_used="ensemble"
        ),
        TestPrediction(
            prediction=1,
            probability=0.85,
            risk_level="HIGH",
            interpretation="High risk due to multiple factors",
            model_used="ensemble"
        )
    ]

@pytest.fixture
def performance_thresholds(environment_config):
    """Get environment-specific performance thresholds."""
    env = os.getenv("DEPLOYMENT_ENV", "development")
    return environment_config["models"][env]["performance_thresholds"]

@pytest.fixture
def api_config(environment_config):
    """Get environment-specific API configuration."""
    env = os.getenv("DEPLOYMENT_ENV", "development")
    return environment_config[env]

@pytest.fixture
def synthetic_dataset():
    """Generate synthetic dataset for testing."""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate features with realistic distributions
    data = {
        "age": np.random.normal(55, 10, n_samples).astype(int),
        "sex": np.random.binomial(1, 0.5, n_samples),
        "cp": np.random.randint(0, 4, n_samples),
        "trestbps": np.random.normal(130, 20, n_samples).astype(int),
        "chol": np.random.normal(220, 40, n_samples).astype(int),
        "fbs": np.random.binomial(1, 0.2, n_samples),
        "restecg": np.random.randint(0, 3, n_samples),
        "thalach": np.random.normal(150, 20, n_samples).astype(int),
        "exang": np.random.binomial(1, 0.3, n_samples),
        "oldpeak": np.random.exponential(1, n_samples),
        "slope": np.random.randint(0, 3, n_samples),
        "ca": np.random.randint(0, 4, n_samples),
        "thal": np.random.randint(1, 4, n_samples)
    }
    
    # Generate labels with dependency on features
    probabilities = 1 / (1 + np.exp(-(
        0.03 * data["age"] +
        0.5 * data["sex"] +
        0.4 * data["cp"] +
        0.01 * (data["trestbps"] - 120) +
        0.005 * (data["chol"] - 200) +
        0.3 * data["exang"] +
        0.4 * data["oldpeak"] +
        0.3 * data["ca"]
    )))
    
    data["target"] = (np.random.random(n_samples) < probabilities).astype(int)
    
    return data

@pytest.fixture
def deployment_validation_data():
    """Configuration for deployment validation."""
    return {
        "required_endpoints": [
            "/health",
            "/predict",
            "/batch_predict",
            "/models/info"
        ],
        "required_methods": ["GET", "POST"],
        "required_response_fields": [
            "prediction",
            "probability",
            "risk_level",
            "interpretation"
        ],
        "required_security_headers": [
            "X-Content-Type-Options",
            "X-Frame-Options",
            "X-XSS-Protection",
            "Strict-Transport-Security"
        ],
        "performance_requirements": {
            "max_response_time": 200,
            "min_requests_per_second": 10
        }
    }

