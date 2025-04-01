# 🫀 Heart Disease Risk API
[![codecov](https://codecov.io/gh/coop-columb/heart-disease-mlp/branch/main/graph/badge.svg)](https://codecov.io/gh/coop-columb/heart-disease-mlp)
## 📦 Project Overview

A FastAPI-based ML inference service for heart disease prediction.

## 🚀 Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn src.heart_api.main:app --reload
