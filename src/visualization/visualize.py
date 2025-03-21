"""
Visualization functions for heart disease prediction project.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from typing import Dict, List, Any, Union, Optional, Tuple


def plot_confusion_matrix(y_true, y_pred, normalize=False, title=None, figsize=(8, 6)):
    """
    Plot confusion matrix for classification results.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        normalize: Whether to normalize the confusion matrix
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        square=True,
        xticklabels=['Negative', 'Positive'],
        yticklabels=['Negative', 'Positive']
    )
    
    # Set labels
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    
    # Set title
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Confusion Matrix')
    
    return fig


def plot_roc_curve(y_true, y_score, title=None, figsize=(8, 6)):
    """
    Plot ROC curve for binary classification results.
    
    Args:
        y_true: True labels
        y_score: Predicted probabilities
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot ROC curve
    ax.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # Set labels and limits
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    
    # Set title
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Receiver Operating Characteristic')
    
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_precision_recall_curve(y_true, y_score, title=None, figsize=(8, 6)):
    """
    Plot precision-recall curve for binary classification results.
    
    Args:
        y_true: True labels
        y_score: Predicted probabilities
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Compute precision-recall curve and average precision
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    avg_precision = average_precision_score(y_true, y_score)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot precision-recall curve
    ax.plot(recall, precision, lw=2, label=f'AP = {avg_precision:.3f}')
    
    # Set labels and limits
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    
    # Set title
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Precision-Recall Curve')
    
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_feature_importance(feature_names, importances, title=None, figsize=(10, 8), top_n=None):
    """
    Plot feature importance for a machine learning model.
    
    Args:
        feature_names: List of feature names
        importances: Array of feature importance scores
        title: Plot title
        figsize: Figure size
        top_n: Number of top features to show (None for all)
        
    Returns:
        Matplotlib figure
    """
    # Create DataFrame of feature importances
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Select top N features if specified
    if top_n is not None:
        importance_df = importance_df.head(top_n)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot horizontal bar chart
    sns.barplot(
        x='Importance', 
        y='Feature', 
        data=importance_df,
        palette='viridis'
    )
    
    # Set labels
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    
    # Set title
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Feature Importance')
    
    ax.grid(True, axis='x', alpha=0.3)
    
    return fig


def plot_prediction_breakdown(patient_data, prediction, feature_contributions, top_n=10, figsize=(12, 8)):
    """
    Create a visual explanation of a model prediction for a patient.
    
    Args:
        patient_data: Dictionary of patient features
        prediction: Model prediction (probability of heart disease)
        feature_contributions: Dictionary mapping features to their contributions
        top_n: Number of top contributing features to display
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Sort feature contributions by absolute value
    sorted_contributions = sorted(
        feature_contributions.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    
    # Select top N features
    top_features = sorted_contributions[:top_n]
    
    # Create DataFrames for plotting
    feature_names = []
    contribution_values = []
    colors = []
    
    for feature, contribution in top_features:
        feature_names.append(feature)
        contribution_values.append(contribution)
        colors.append('green' if contribution > 0 else 'red')
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot waterfall chart
    bars = ax.barh(feature_names, contribution_values, color=colors)
    
    # Add patient values as text
    for i, feature in enumerate(feature_names):
        if feature in patient_data:
            ax.text(
                0, i, 
                f" Value: {patient_data[feature]}", 
                va='center',
                fontweight='bold'
            )
    
    # Set labels
    ax.set_xlabel('Contribution to Prediction')
    ax.set_ylabel('Feature')
    
    # Set title
    risk_percentage = prediction * 100
    title = f"Patient's Heart Disease Risk: {risk_percentage:.1f}%\n"
    title += "Green = Increases Risk, Red = Decreases Risk"
    ax.set_title(title)
    
    ax.grid(True, axis='x', alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    
    return fig