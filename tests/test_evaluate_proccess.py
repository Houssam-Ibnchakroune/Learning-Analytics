from src.models.evaluate import (
    evaluate_classification,
    plot_confusion_matrix,
    plot_roc_curve,
    get_feature_importance,
    plot_feature_importance,
    evaluate_model_complete
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

# Charger les métadonnées du modèle pour récupérer les noms de features
logger.info("Chargement des métadonnées...")
with open('models/model_metadata_day180_test.json', 'r') as f:
    model_metadata = json.load(f)
    feature_names = model_metadata['feature_names']

# Charger les données de test
logger.info("Chargement des données de test...")
X_test = np.load('data/processed/X_features.npy')[:1000]  # Prendre un échantillon pour test
y_test = np.load('data/processed/y_labels.npy')[:1000]

# Faire les prédictions
logger.info("Génération des prédictions...")
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

logger.info("=" * 60)
logger.info("TEST DU MODULE evaluate.py")
logger.info("=" * 60)

# Test 1: Évaluation des métriques de classification
logger.info("\n Test de evaluate_classification()...")
metrics = evaluate_classification(y_test, y_pred, class_names=['Pass', 'Fail'])
logger.info(" Métriques calculées avec succès")

# Test 2: Matrice de confusion
logger.info("\n Test de plot_confusion_matrix()...")
plot_confusion_matrix(y_test, y_pred, 
                     save_path='reports/figures/confusion_matrix_test.png',
                     class_names=['Pass', 'Fail'])
logger.info(" Matrice de confusion générée")

# Test 3: Courbe ROC
logger.info("\n Test de plot_roc_curve()...")
auc_score = plot_roc_curve(y_test, y_proba, 
                          save_path='reports/figures/roc_curve_test.png')
logger.info(f" Courbe ROC générée (AUC = {auc_score:.4f})")

# Test 4: Feature importance
logger.info("\n Test de get_feature_importance()...")
importance_df = get_feature_importance(model, feature_names)
logger.info(" Feature importance calculée")

# Test 5: Visualisation des features importantes
logger.info("\n Test de plot_feature_importance()...")
plot_feature_importance(importance_df, top_n=15, 
                       save_path='reports/figures/feature_importance_test.png')
logger.info(" Feature importance visualisée")

# Test 6: Évaluation complète
logger.info("\n Test de evaluate_model_complete()...")
results = evaluate_model_complete(
    model, X_test, y_test, y_proba=y_proba,
    feature_names=feature_names,
    save_dir='reports/figures'
)
logger.info(" Évaluation complète terminée")

logger.info("\n" + "=" * 60)
logger.info(" TOUS LES TESTS RÉUSSIS!")
logger.info("=" * 60)
logger.info(f"\n Résultats finaux:")
logger.info(f"   • Accuracy:  {results['accuracy']:.4f}")
logger.info(f"   • Precision: {results['precision']:.4f}")
logger.info(f"   • Recall:    {results['recall']:.4f}")
logger.info(f"   • F1-Score:  {results['f1_score']:.4f}")
logger.info(f"   • AUC:       {results['auc']:.4f}")
logger.info("\nModule evaluate.py validé et prêt à l'emploi! ")
