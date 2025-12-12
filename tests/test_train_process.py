from datetime import datetime
from src.models.train import save_model_metadata, train_random_forest, train_with_cross_validation, save_model, load_model
import json
import numpy as np
from utils.login_setup import setup_logging
import logging
setup_logging()
logger = logging.getLogger(__name__)

# Charger les données (exemple)
X_train = np.load('data/processed/X_features.npy')
y_train = np.load('data/processed/y_labels.npy')

# Charger les métadonnées des features
metadata_path = 'data/processed/features_metadata.json'



# Configuration du modèle
config = {
    'n_estimators': 200,
    'max_features': 'sqrt',
    'min_samples_split': 10,
    'random_state': 42
}

# Entraînement avec cross-validation
model, cv_scores, model_metadata = train_with_cross_validation(X_train, y_train, metadata_path, config, n_splits=5)


# Sauvegarde
save_model(model, 'models/random_forest_day180_test.pkl')
save_model_metadata(model_metadata, 'models/model_metadata_day180_test.json')

logger.info("Module train.py prêt à l'emploi !")