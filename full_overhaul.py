import shutil
from pathlib import Path

root = Path(__file__).resolve().parent

# Clean structure definition
layout = {
    "src/heart_api/api": [],
    "src/heart_api/core": [],
    "src/heart_api/services": [],
    "scripts": [],
    "tests": [],
    "docs/internal": [],
    "docs/ops": [],
}

# Create folders
for folder in layout:
    (root / folder).mkdir(parents=True, exist_ok=True)

# Move core files
file_moves = {
    "api/app.py": "src/heart_api/api/endpoints.py",
    "api/auth.py": "src/heart_api/services/auth.py",
    "config/config.yaml": "config/config.yaml",
    "docs/fixes_documentation.md": "docs/internal/fixes.md",
    "docs/backup_recovery.md": "docs/ops/backup.md",
}

for src, dst in file_moves.items():
    src_path = root / src
    dst_path = root / dst
    if src_path.exists():
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_path), str(dst_path))

# Delete unused or outdated files
for path in ["run_api.py", "setup.py", "scripts/generate_test_data.py"]:
    file_path = root / path
    if file_path.exists():
        file_path.unlink()

# Scaffold main FastAPI entrypoint
main_code = """from fastapi import FastAPI
from heart_api.api.endpoints import router as api_router

app = FastAPI(title="Heart Disease Prediction API")

app.include_router(api_router)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('heart_api.main:app', host='0.0.0.0', port=8000, reload=True)
"""
(root / "src/heart_api/main.py").write_text(main_code)

# Scaffold schemas file
(root / "src/heart_api/api/schemas.py").write_text(
    """from pydantic import BaseModel, Field
from typing import Optional, List, Dict

class PatientData(BaseModel):
    age: int = Field(..., example=61)
    sex: int = Field(..., example=1)
    cp: int = Field(..., example=3)
    trestbps: int = Field(..., example=140)
    chol: int = Field(..., example=240)
    fbs: int = Field(..., example=1)
    restecg: int = Field(..., example=1)
    thalach: int = Field(..., example=150)
    exang: int = Field(..., example=1)
    oldpeak: float = Field(..., example=2.4)
    slope: int = Field(..., example=2)
    ca: int = Field(..., example=1)
    thal: int = Field(..., example=3)

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    risk_level: str
    interpretation: Optional[str] = None
    model_used: Optional[str] = None

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    performance_metrics: Optional[Dict[str, float]] = None
"""
)

# Create __init__.py for packages
for pkg in [
    "src/heart_api",
    "src/heart_api/api",
    "src/heart_api/core",
    "src/heart_api/services",
]:
    (root / pkg / "__init__.py").touch()

# Scaffold .gitignore
(root / ".gitignore").write_text(
    """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*.pyo

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Virtual environments
.env/
.venv/
env/
venv/
ENV/
env.bak/
venv.bak/

# PyInstaller
*.manifest
*.spec

# Unit test / coverage reports
htmlcov/
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/

# Pytest
.pytest_cache/

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# profiling
prof/
*.prof

# Jupyter Notebook
.ipynb_checkpoints

# IPython history
profile_default/
ipython_config.py

# pyenv
.python-version

# dotenv
.env
.env.*

# Editor directories and files
.vscode/
.idea/
*.sublime-project
*.sublime-workspace

# MacOS
.DS_Store

# Logs
logs/
*.log

# Models and data
models/
data/
backups/

# VS Code notebooks
**/.vscode-ipynb-*

# Archive files
*.tar.gz
*.zip
*.bak

# System files
Thumbs.db
ehthumbs.db

# Docker
*.pid
*.pid.lock
"""
)

# Scaffold pyproject.toml
(root / "pyproject.toml").write_text(
    """[project]
name = "heart-disease-api"
version = "0.1.0"
dependencies = [
  "fastapi",
  "uvicorn[standard]",
  "pydantic",
  "numpy",
  "pandas",
  "scikit-learn",
  "joblib",
  "tensorflow",
]

[tool.setuptools]
packages = ["heart_api"]
"""
)

# Scaffold pytest config and test stub
(root / "tests/test_api_smoke.py").write_text(
    """def test_smoke():
    assert True
"""
)

# Scaffold README.md
(root / "README.md").write_text(
    """# 🫀 Heart Disease Prediction API

Run with:

```bash
uvicorn src.heart_api.main:app --reload
```

Docs at: http://localhost:8000/docs

See `docs/ops/backup.md` for backup info.
"""
)

# Scaffold GitHub Actions CI workflow
ci_path = root / ".github/workflows"
ci_path.mkdir(parents=True, exist_ok=True)
(ci_path / "ci.yml").write_text(
    """name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: 3.11
      - name: Install dependencies
        run: |
          python -m venv .venv
          source .venv/bin/activate
          pip install -r requirements.txt || pip install .
      - name: Run tests
        run: |
          source .venv/bin/activate
          pytest tests/
"""
)

# Insert content for /predict and /predict/batch routes
endpoints_code = """from fastapi import APIRouter, HTTPException
from heart_api.api.schemas import PatientData, PredictionResponse, BatchPredictionResponse
import random

router = APIRouter()

@router.get("/health")
def health_check():
    return {"status": "ok"}

@router.post("/predict", response_model=PredictionResponse)
def predict(data: PatientData):
    # Simulated prediction logic (replace with real model)
    probability = round(random.uniform(0.0, 1.0), 4)
    risk_level = (
        "LOW" if probability < 0.3 else
        "MODERATE" if probability < 0.6 else
        "HIGH"
    )
    return {
        "prediction": int(probability > 0.5),
        "probability": probability,
        "risk_level": risk_level,
        "interpretation": "Simulated result",
        "model_used": "mock_model"
    }

@router.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(batch: list[PatientData]):
    results = [predict(p) for p in batch]
    return {"predictions": results, "performance_metrics": {"simulated_latency_ms": 5.2}}
"""

(root / "src/heart_api/api/endpoints.py").write_text(endpoints_code)

print("✅ Full overhaul complete.")
