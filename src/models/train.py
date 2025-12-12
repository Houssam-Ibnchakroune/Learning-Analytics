"""
Module pour l'entraînement des modèles ML.
"""
import json
from datetime import datetime
import logging
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score

logger = logging.getLogger(__name__)
def train_random_forest(X_train, y_train, config=None):
    """
    Entraîne un modèle Random Forest.
    
    Args:
        X_train: Features d'entraînement (numpy array)
        y_train: Labels (numpy array)
        config: Dictionnaire avec paramètres du modèle (optionnel)
                {
                    'n_estimators': 200,
                    'max_features': 'sqrt',
                    'min_samples_split': 10,
                    'random_state': 42
                }
        
    Returns:
        model: Modèle RandomForestClassifier entraîné
    """
    logger.info("Entraînement du modèle Random Forest...")
    if config is None:
        config = {
            'n_estimators': 200,
            'max_features': 'sqrt',
            'min_samples_split': 10,
            'random_state': 42
        }
    
    model = RandomForestClassifier(**config)
    model.fit(X_train, y_train)
    logger.info("Modèle entraîné avec succès.")
    
    return model


def train_with_cross_validation(X_train, y_train,feature_metadata, config=None, n_splits=5):
    """
    Entraîne un modèle Random Forest avec cross-validation.
    
    Args:
        X_train: Features d'entraînement
        y_train: Labels
        config: Paramètres du modèle (optionnel)
        n_splits: Nombre de folds pour K-Fold (défaut: 5)
        
    Returns:
        model: Modèle entraîné sur l'ensemble complet
        cv_scores: Liste des scores de validation croisée
    """
    logger.info("Entraînement avec cross-validation...")
    if config is None:
        config = {
            'n_estimators': 200,
            'max_features': 'sqrt',
            'min_samples_split': 10,
            'random_state': 42
        }
    
    model = RandomForestClassifier(**config)
    
    # Cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = []
    
    logger.info(f"Cross-validation avec {n_splits} folds...")
    for fold, (train_index, test_index) in enumerate(kf.split(X_train), 1):
        X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
        y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
        
        model.fit(X_train_fold, y_train_fold)
        score = model.score(X_test_fold, y_test_fold)
        cv_scores.append(score)
        logger.info(f"  Fold {fold}: Accuracy = {score:.4f}")
    
    logger.info(f"\nMoyenne CV: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
    
    # Entraînement final sur l'ensemble complet
    model.fit(X_train, y_train)
    
    with open(feature_metadata, 'r') as f:
        metadata = json.load(f)
        column_names = np.array(metadata['column_names'])
        encode_dict = metadata['encode_dict']
        
    model_metadata = {
    'model_type': 'RandomForestClassifier',
    'n_estimators': model.n_estimators,
    'max_features': model.max_features,
    'min_samples_split': model.min_samples_split,
    'n_features': X_train.shape[1],
    'feature_names': column_names.tolist() if hasattr(column_names, 'tolist') else list(column_names),
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'training_samples': X_train.shape[0]
}
    
    return model, cv_scores, model_metadata


def save_model(model, filepath):
    """
    Sauvegarde un modèle entraîné.
    
    Args:
        model: Modèle scikit-learn
        filepath: Chemin de sauvegarde (ex: 'models/random_forest_day180.pkl')
    """
    joblib.dump(model, filepath)
    logger.info(f"Modèle sauvegardé : {filepath}")


def save_model_metadata(metadata: dict, filepath: str):
    """
    Sauvegarde les métadonnées du modèle dans un fichier JSON.
    
    Args:
        metadata: Dictionnaire des métadonnées
        filepath: Chemin du fichier JSON (ex: 'models/model_metadata_day180.json')
    """
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Métadonnées du modèle sauvegardées : {filepath}")


def load_model(filepath):
    """
    Charge un modèle sauvegardé.
    
    Args:
        filepath: Chemin du modèle (ex: 'models/random_forest_day180.pkl')
        
    Returns:
        model: Modèle chargé
    """
    model = joblib.load(filepath)
    logger.info(f"Modèle chargé : {filepath}")
    return model


# Exemple d'utilisation
if __name__ == "__main__":
    import sys
    sys.path.append('..')
    
    # Charger les données (exemple)
    # X_train = np.load('../data/processed/X_train.npy')
    # y_train = np.load('../data/processed/y_train.npy')
    
    # Configuration du modèle
    config = {
        'n_estimators': 200,
        'max_features': 'sqrt',
        'min_samples_split': 10,
        'random_state': 42
    }
    
    # Entraînement avec cross-validation
    # model, cv_scores = train_with_cross_validation(X_train, y_train, config, n_splits=5)
    
    # Sauvegarde
    # save_model(model, '../models/random_forest_day180.pkl')
    
    logger.info("Module train.py prêt à l'emploi !")