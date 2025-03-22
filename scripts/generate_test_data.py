#!/usr/bin/env python3
"""
Generate synthetic data for CI/CD tests.
This script creates minimal test data for testing the heart disease prediction model.
"""

import os

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Create synthetic data
X = np.random.rand(100, 13)
y = np.random.randint(0, 2, 100)

# Create preprocessor
preprocessor = StandardScaler()
preprocessor.fit(X)

# Save data and preprocessor
os.makedirs("data/processed", exist_ok=True)
np.savez(
    "data/processed/processed_data.npz",
    X_train=X[:60],
    y_train=y[:60],
    X_val=X[60:80],
    y_val=y[60:80],
    X_test=X[80:],
    y_test=y[80:],
)
joblib.dump(preprocessor, "data/processed/preprocessor.joblib")

# Create original splits
splits = {
    "train_indices": np.arange(60),
    "val_indices": np.arange(60, 80),
    "test_indices": np.arange(80, 100),
}
joblib.dump(splits, "data/processed/original_splits.joblib")

# Create some info for metadata
with open("data/processed/processing_metadata.txt", "w") as f:
    f.write("Synthetic data created for testing\n")
    f.write("Features: 13, Samples: 100\n")
    f.write("Train: 60, Val: 20, Test: 20\n")

print("✅ Synthetic test data created successfully")
