# src/models/__init__.py
"""
Module pour les modèles ML.
"""

from .train import train_random_forest, train_with_cross_validation, save_model, load_model
from .predict import predict, predict_proba, predict_with_risk_score, predict_batch
from .evaluate import (
    evaluate_classification, 
    plot_confusion_matrix, 
    plot_roc_curve,
    get_feature_importance,
    plot_feature_importance,
    evaluate_model_complete
)

__all__ = [
    # Train
    'train_random_forest',
    'train_with_cross_validation',
    'save_model',
    'load_model',
    # Predict
    'predict',
    'predict_proba',
    'predict_with_risk_score',
    'predict_batch',
    # Evaluate
    'evaluate_classification',
    'plot_confusion_matrix',
    'plot_roc_curve',
    'get_feature_importance',
    'plot_feature_importance',
    'evaluate_model_complete'
]
