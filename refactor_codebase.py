import os
import shutil
from pathlib import Path

root = Path(__file__).resolve().parent

# Define all target directories
dirs_to_create = [
    "src/heart_api/api",
    "src/heart_api/core",
    "src/heart_api/services",
    "tests",
]

# Create directories
for d in dirs_to_create:
    os.makedirs(root / d, exist_ok=True)

# Create pyproject.toml file
pyproject_path = root / "pyproject.toml"
pyproject_content = """[project]
name = "heart-disease-mlp"
version = "0.1.0"
description = "Heart Disease Prediction API with ML Pipeline"
requires-python = ">=3.8"
dependencies = [
    "numpy>=1.18.0",
    "pandas>=1.0.0",
    "scikit-learn>=0.24.0",
    "tensorflow>=2.4.0",
    "fastapi>=0.68.0",
    "uvicorn>=0.15.0"
]

[tool.setuptools]
packages = ["heart_api"]

[tool.black]
line-length = 100
target-version = ['py38', 'py39', 'py310']
include = '\\.pyi?$'

[tool.isort]
profile = "black"
line_length = 100

[tool.pytest.ini_options]
testpaths = ["tests"]
norecursedirs = ["scripts"]
"""
pyproject_path.write_text(pyproject_content)

# Move and rename critical files
file_moves = {
    "api/app.py": "src/heart_api/api/endpoints.py",
    "api/auth.py": "src/heart_api/services/auth.py",
    "config/config.yaml": "config/config.yaml",  # unchanged
    "docs/fixes_documentation.md": "docs/internal/fixes.md",
    "docs/backup_recovery.md": "docs/ops/backup.md",
}

for src, dst in file_moves.items():
    src_path = root / src
    dst_path = root / dst
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if src_path.exists():
        shutil.move(str(src_path), str(dst_path))

# Delete unnecessary or broken files
for f in ["run_api.py", "setup.py", "scripts/generate_test_data.py"]:
    path = root / f
    if path.exists():
        path.unlink()

# Scaffold new entrypoint
main_path = root / "src/heart_api/main.py"
main_path.write_text(
    """from fastapi import FastAPI
from heart_api.api.endpoints import router as api_router

app = FastAPI(title="Heart Disease Prediction API")

app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("heart_api.main:app", host="0.0.0.0", port=8000, reload=True)
"""
)

# Touch __init__.py files
for package_dir in [
    "src/heart_api",
    "src/heart_api/api",
    "src/heart_api/core",
    "src/heart_api/services",
]:
    init_file = root / package_dir / "__init__.py"
    init_file.touch()

print("✅ Refactor complete.")
