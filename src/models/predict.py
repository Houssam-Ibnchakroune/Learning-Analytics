"""
Module pour les prédictions avec modèles entraînés.
"""

import numpy as np
import pandas as pd
import joblib


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


def predict(model, X_test):
    """
    Fait des prédictions avec le modèle.
    
    Args:
        model: Modèle scikit-learn entraîné
        X_test: Features de test (numpy array ou DataFrame)
        
    Returns:
        predictions: Array numpy avec les prédictions (0 = Pass, 1 = Fail)
    """
    predictions = model.predict(X_test)
    return predictions


def predict_proba(model, X_test):
    """
    Retourne les probabilités de prédiction.
    
    Args:
        model: Modèle scikit-learn entraîné
        X_test: Features de test
        
    Returns:
        probabilities: Array numpy avec probabilités pour chaque classe
                      Shape: (n_samples, n_classes)
                      Colonne 0 = prob(Pass), Colonne 1 = prob(Fail)
    """
    probabilities = model.predict_proba(X_test)
    return probabilities


def predict_with_risk_score(model, X_test):
    """
    Fait des prédictions avec score de risque (probabilité d'échec).
    
    Args:
        model: Modèle entraîné
        X_test: Features de test
        
    Returns:
        predictions: Prédictions (0 = Pass, 1 = Fail)
        risk_scores: Scores de risque (0-1, 1 = risque élevé d'échec)
    """
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    # Score de risque = probabilité de la classe "Fail" (indice 1)
    risk_scores = probabilities[:, 1]
    
    return predictions, risk_scores


def predict_batch(model, data_path, output_path=None):
    """
    Fait des prédictions sur un batch de données depuis un fichier.
    
    Args:
        model: Modèle entraîné
        data_path: Chemin vers fichier CSV ou NPY avec features
        output_path: Chemin pour sauvegarder résultats (optionnel)
        
    Returns:
        results_df: DataFrame avec prédictions et scores de risque
    """
    # Charger les données
    if data_path.endswith('.npy'):
        X = np.load(data_path)
    elif data_path.endswith('.csv'):
        X = pd.read_csv(data_path).values
    else:
        raise ValueError("Format non supporté. Utilisez .npy ou .csv")
    
    # Prédictions
    predictions, risk_scores = predict_with_risk_score(model, X)
    
    # Créer DataFrame avec résultats
    results_df = pd.DataFrame({
        'prediction': predictions,
        'risk_score': risk_scores,
        'risk_level': pd.cut(risk_scores, 
                            bins=[0, 0.3, 0.7, 1.0], 
                            labels=['Faible', 'Moyen', 'Élevé'])
    })
    
    # Sauvegarder si chemin fourni
    if output_path:
        results_df.to_csv(output_path, index=False)
        print(f"✅ Prédictions sauvegardées : {output_path}")
    
    return results_df


def classify_risk_level(risk_score):
    """
    Classe le niveau de risque basé sur le score.
    
    Args:
        risk_score: Score de risque (0-1)
        
    Returns:
        risk_level: 'Faible', 'Moyen', ou 'Élevé'
    """
    if risk_score < 0.3:
        return 'Faible'
    elif risk_score < 0.7:
        return 'Moyen'
    else:
        return 'Élevé'


# Exemple d'utilisation
if __name__ == "__main__":
    import sys
    sys.path.append('..')
    
    # Charger modèle
    # model = load_model('../models/random_forest_day180.pkl')
    
    # Charger données de test
    # X_test = np.load('../data/processed/X_test.npy')
    
    # Prédictions simples
    # predictions = predict(model, X_test)
    # print(f"Prédictions : {predictions[:10]}")
    
    # Prédictions avec scores de risque
    # predictions, risk_scores = predict_with_risk_score(model, X_test)
    # print(f"Scores de risque : {risk_scores[:10]}")
    
    print("Module predict.py prêt à l'emploi !")
