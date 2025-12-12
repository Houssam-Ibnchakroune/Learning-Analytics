from src.models.predict import (
    predict,
    predict_proba,
    predict_with_risk_score,
    predict_batch,
    categorize_risk
)
from src.models.train import load_model
import numpy as np
import json
from utils.login_setup import setup_logging
import logging

setup_logging()
logger = logging.getLogger(__name__)

# Charger le modèle entraîné
logger.info("Chargement du modèle...")
model = load_model('models/random_forest_day180_test.pkl')

# Charger les métadonnées du modèle
logger.info("Chargement des métadonnées...")
with open('models/model_metadata_day180_test.json', 'r') as f:
    model_metadata = json.load(f)
    feature_names = model_metadata['feature_names']
    logger.info(f"Modèle entraîné avec {model_metadata['n_features']} features")

# Charger données de test (échantillon)
logger.info("Chargement des données de test...")
X_test = np.load('data/processed/X_features.npy')[:100]
y_test = np.load('data/processed/y_labels.npy')[:100]

logger.info("=" * 60)
logger.info("TEST DU MODULE predict.py")
logger.info("=" * 60)

# Test 1: Prédictions simples
logger.info("\n Test de predict()...")
predictions = predict(model, X_test)
logger.info(f"   Nombre de prédictions: {len(predictions)}")
logger.info(f"   Échantillon (10 premiers): {predictions[:10]}")
logger.info(f"   Distribution: Pass={np.sum(predictions==0)}, Fail={np.sum(predictions==1)}")
logger.info(" Prédictions simples OK")

# Test 2: Probabilités de prédiction
logger.info("\n Test de predict_proba()...")
probabilities = predict_proba(model, X_test)
logger.info(f"   Shape des probabilités: {probabilities.shape}")
logger.info(f"   Échantillon (5 premiers):")
for i in range(5):
    logger.info(f"      Exemple {i+1}: Pass={probabilities[i,0]:.3f}, Fail={probabilities[i,1]:.3f}")
logger.info(" Probabilités calculées OK")

# Test 3: Prédictions avec scores de risque
logger.info("\n Test de predict_with_risk_score()...")
predictions, risk_scores = predict_with_risk_score(model, X_test)
logger.info(f"   Scores de risque (10 premiers): {risk_scores[:10]}")
logger.info(f"   Risque moyen: {np.mean(risk_scores):.3f}")
logger.info(f"   Risque min: {np.min(risk_scores):.3f}, max: {np.max(risk_scores):.3f}")
logger.info(" Scores de risque calculés OK")

# Test 4: Catégorisation des risques
logger.info("\n Test de categorize_risk()...")
risk_categories = categorize_risk(risk_scores)
unique, counts = np.unique(risk_categories, return_counts=True)
logger.info(f"   Distribution des catégories de risque:")
for cat, count in zip(unique, counts):
    logger.info(f"      {cat}: {count} étudiants ({count/len(risk_categories)*100:.1f}%)")
logger.info(" Catégorisation des risques OK")

# Test 5: Comparaison avec vraies valeurs
logger.info("\n Test de comparaison prédictions vs réalité...")
correct = np.sum(predictions == y_test)
accuracy = correct / len(y_test)
logger.info(f"   Prédictions correctes: {correct}/{len(y_test)}")
logger.info(f"   Accuracy sur échantillon: {accuracy:.2%}")

# Analyse des erreurs
false_positives = np.sum((predictions == 1) & (y_test == 0))
false_negatives = np.sum((predictions == 0) & (y_test == 1))
logger.info(f"   Faux positifs (prédit Fail, vrai Pass): {false_positives}")
logger.info(f"   Faux négatifs (prédit Pass, vrai Fail): {false_negatives}")
logger.info(" Comparaison effectuée")

# Test 6: Prédiction batch
logger.info("\n Test de predict_batch()...")
results_df = predict_batch(model, 'data/processed/X_features.npy', 
                          output_path='reports/predictions_test.csv')
logger.info(f"   Nombre de prédictions batch: {len(results_df)}")
logger.info(f"   Colonnes du résultat: {list(results_df.columns)}")
logger.info(" Prédiction batch OK")

# Résumé final
logger.info("\n" + "=" * 60)
logger.info(" TOUS LES TESTS RÉUSSIS!")
logger.info("=" * 60)
logger.info("\n Résumé des résultats:")
logger.info(f"   • Échantillon testé: {len(X_test)} étudiants")
logger.info(f"   • Accuracy: {accuracy:.2%}")
logger.info(f"   • Risque moyen: {np.mean(risk_scores):.1%}")
logger.info(f"   • Faux positifs: {false_positives}")
logger.info(f"   • Faux négatifs: {false_negatives}")
logger.info("\n Fichiers générés:")
logger.info("   • reports/predictions_test.csv")
logger.info("\nModule predict.py validé et prêt à l'emploi! ")