#!/usr/bin/env python3
"""
Generate synthetic models for CI/CD tests.
This script creates minimal test models for testing the heart disease prediction API.
"""

import os

import joblib
import numpy as np

# tensorflow is imported by keras
from sklearn.neural_network import MLPClassifier
from tensorflow import keras

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Create minimal sklearn model
X = np.random.rand(100, 13)
y = np.random.randint(0, 2, 100)
sklearn_model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=10)
sklearn_model.fit(X, y)
joblib.dump(sklearn_model, "models/sklearn_mlp_model.joblib")
print("✅ Scikit-learn model created successfully")

# Create minimal keras model
keras_model = keras.Sequential(
    [
        keras.layers.Dense(10, activation="relu", input_shape=(13,)),
        keras.layers.Dense(1, activation="sigmoid"),
    ]
)
keras_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
keras_model.fit(X, y, epochs=1, verbose=0)
keras_model.save("models/keras_mlp_model.h5")
print("✅ Keras model created successfully")
