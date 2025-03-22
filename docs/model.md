# Heart Disease Prediction Model Architecture

This document provides a detailed explanation of the machine learning models used in the Heart Disease Prediction System.

## Table of Contents

- [Overview](#overview)
- [Data Preprocessing](#data-preprocessing)
- [Scikit-learn MLP Architecture](#scikit-learn-mlp-architecture)
- [Keras MLP Architecture](#keras-mlp-architecture)
- [Ensemble Approach](#ensemble-approach)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Evaluation Metrics](#evaluation-metrics)
- [Feature Importance](#feature-importance)
- [Model Interpretation](#model-interpretation)
- [Model Persistence](#model-persistence)

## Overview

The Heart Disease Prediction System uses two complementary Multi-Layer Perceptron (MLP) neural networks:
1. A scikit-learn MLPClassifier implementation
2. A Keras/TensorFlow deep learning implementation

These models are then combined in an ensemble to provide more robust predictions.

## Data Preprocessing

Before training the models, the data undergoes several preprocessing steps:

1. **Missing Value Imputation**:
   - Numerical features: imputed with median
   - Categorical features: imputed with most frequent value

2. **Feature Scaling**:
   - Standardization (z-score normalization) for numerical features

3. **Feature Engineering**:
   - Age groups (age brackets for risk stratification)
   - Blood pressure categories (normal, elevated, hypertension)
   - Cholesterol risk levels
   - BMI estimation using available parameters

4. **Train-Test Split**:
   - Stratified splitting to maintain class balance
   - 70% training, 15% validation, 15% testing

The preprocessing pipeline is implemented using scikit-learn's `Pipeline` and `ColumnTransformer` to ensure consistent preprocessing of new data during inference.

## Scikit-learn MLP Architecture

The scikit-learn MLP model uses the following configuration:

```python
MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    batch_size='auto',
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=20,
    random_state=42
)
```

Key components:
- **Hidden Layers**: Three layers with 128, 64, and 32 neurons respectively
- **Activation Function**: ReLU (Rectified Linear Unit)
- **Optimizer**: Adam with adaptive learning rate
- **Regularization**: L2 regularization with alpha=0.0001
- **Early Stopping**: Enabled to prevent overfitting
- **Batch Size**: Auto (min(200, n_samples))

## Keras MLP Architecture

The Keras MLP model has a more complex architecture with additional regularization techniques:

```python
model = keras.Sequential([
    keras.layers.Input(shape=(n_features,)),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
)
```

Key components:
- **Input Layer**: Matches the number of features in the dataset
- **Hidden Layers**: Three fully connected layers with 128, 64, and 32 neurons
- **Activation Function**: ReLU for hidden layers, Sigmoid for output layer
- **Regularization**: Dropout layers with rates of 0.3, 0.2, and 0.1
- **Normalization**: BatchNormalization after input and between hidden layers
- **Output Layer**: Single neuron with sigmoid activation for binary classification
- **Loss Function**: Binary cross-entropy
- **Optimizer**: Adam with learning rate of 0.001
- **Metrics**: Accuracy and Area Under ROC Curve (AUC)

The Keras model is trained using the following callbacks:
- Early stopping (monitors validation loss with patience=30)
- Model checkpoint (saves the best model based on validation loss)
- Learning rate reduction (reduces learning rate when progress plateaus)

## Ensemble Approach

The ensemble combines predictions from both models to create a more robust classifier:

1. **Probability Averaging**:
   ```python
   ensemble_prob = 0.6 * keras_prob + 0.4 * sklearn_prob
   ```

2. **Weighted Voting**: The weights are determined by each model's performance on the validation set.

3. **Threshold Calibration**: The classification threshold is calibrated using the validation set to optimize F1 score.

The ensemble helps to:
- Reduce variance and avoid overfitting
- Improve prediction stability
- Balance the strengths of both modeling approaches

## Hyperparameter Tuning

Both models undergo hyperparameter tuning using Optuna, an efficient hyperparameter optimization framework:

### Scikit-learn MLP Tuning

Parameters optimized:
- Hidden layer sizes
- Alpha (L2 regularization term)
- Learning rate
- Batch size
- Activation function

### Keras MLP Tuning

Parameters optimized:
- Number of hidden layers (2-4)
- Number of units in each layer
- Dropout rates
- Learning rate
- Batch size
- Optimizer selection

### Tuning Process

1. **Objective Function**: Maximize ROC AUC on validation set
2. **Cross-Validation**: 5-fold stratified cross-validation
3. **Search Algorithm**: Tree-structured Parzen Estimator (TPE)
4. **Number of Trials**: 100 for each model
5. **Pruning**: Enables early stopping of unpromising trials

Optuna studies are stored in `models/optuna/` for reference and visualization.

## Evaluation Metrics

The models are evaluated using multiple metrics to provide a comprehensive assessment:

1. **Accuracy**: Proportion of correct predictions
2. **Precision**: Proportion of positive identifications that were actually correct
3. **Recall**: Proportion of actual positives that were identified correctly
4. **F1 Score**: Harmonic mean of precision and recall
5. **ROC AUC**: Area under the Receiver Operating Characteristic curve
6. **PR AUC**: Area under the Precision-Recall curve
7. **Confusion Matrix**: Detailed breakdown of true/false positives/negatives

Current performance metrics for the models:

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|-------|----------|-----------|--------|----------|---------|
| Scikit-learn MLP | 85.3% | 0.858 | 0.841 | 0.849 | 0.929 |
| Keras MLP | 85.3% | 0.845 | 0.870 | 0.857 | 0.926 |
| Ensemble | 86.4% | 0.861 | 0.870 | 0.865 | 0.930 |

## Feature Importance

Feature importance is calculated using multiple techniques:

1. **Permutation Importance**: Measures the decrease in model performance when a feature is randomly shuffled.

2. **SHAP (SHapley Additive exPlanations)**: Calculates the contribution of each feature to the prediction for individual instances.

3. **Partial Dependence Plots**: Show the marginal effect of features on the predicted outcome.

The most important features according to permutation importance are:
1. Chest pain type (cp)
2. Number of major vessels colored by fluoroscopy (ca)
3. Exercise-induced angina (exang)
4. ST depression induced by exercise (oldpeak)
5. Maximum heart rate achieved (thalach)

## Model Interpretation

The system provides clinical interpretation of model predictions based on:

1. **Risk Level Categorization**:
   - Low Risk: Probability < 0.3
   - Moderate Risk: Probability 0.3-0.7
   - High Risk: Probability > 0.7

2. **Feature Contribution Analysis**:
   - Identifies the top features contributing to a specific prediction
   - Provides clinical context for each contributing factor

3. **Confidence Assessment**:
   - Evaluates prediction confidence based on model agreement
   - Considers prediction stability across ensemble members

## Model Persistence

The trained models are persisted to disk for later use in prediction:

1. **Scikit-learn Model**:
   - Stored using joblib serialization
   - File: `models/sklearn_mlp_model.joblib`

2. **Keras Model**:
   - Stored using HDF5 format
   - File: `models/keras_mlp_model.h5`

3. **Preprocessor**:
   - Stored using joblib serialization
   - File: `data/processed/preprocessor.joblib`

4. **Evaluation Results**:
   - Stored using joblib serialization
   - File: `models/evaluation_results.joblib`

These serialized models are loaded by the prediction module for inference during API calls and CLI usage.
