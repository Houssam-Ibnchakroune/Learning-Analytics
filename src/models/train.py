"""
Module pour l'entraînement des modèles ML.
"""

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score


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
    if config is None:
        config = {
            'n_estimators': 200,
            'max_features': 'sqrt',
            'min_samples_split': 10,
            'random_state': 42
        }
    
    model = RandomForestClassifier(**config)
    model.fit(X_train, y_train)
    
    return model


def train_with_cross_validation(X_train, y_train, config=None, n_splits=5):
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
    
    print(f"🔄 Cross-validation avec {n_splits} folds...")
    for fold, (train_index, test_index) in enumerate(kf.split(X_train), 1):
        X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
        y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
        
        model.fit(X_train_fold, y_train_fold)
        score = model.score(X_test_fold, y_test_fold)
        cv_scores.append(score)
        print(f"  Fold {fold}: Accuracy = {score:.4f}")
    
    print(f"\n✅ Moyenne CV: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
    
    # Entraînement final sur l'ensemble complet
    model.fit(X_train, y_train)
    
    return model, cv_scores


def save_model(model, filepath):
    """
    Sauvegarde un modèle entraîné.
    
    Args:
        model: Modèle scikit-learn
        filepath: Chemin de sauvegarde (ex: 'models/random_forest_day180.pkl')
    """
    joblib.dump(model, filepath)
    print(f"✅ Modèle sauvegardé : {filepath}")


def load_model(filepath):
    """
    Charge un modèle sauvegardé.
    
    Args:
        filepath: Chemin du modèle (ex: 'models/random_forest_day180.pkl')
        
    Returns:
        model: Modèle chargé
    """
    model = joblib.load(filepath)
    print(f"✅ Modèle chargé : {filepath}")
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
    
    print("Module train.py prêt à l'emploi !")