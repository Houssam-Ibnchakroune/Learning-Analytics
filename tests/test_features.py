from src.features.build_features import save_features, load_features,engineer_features
from src.data.extract import load_oulad_data
from src.data.transform import prepare_dataset
from utils.login_setup import setup_logging
setup_logging()
import logging
logger = logging.getLogger(__name__)
logger.info("Test du module build_features.py\n")

try:
    # Charger et préparer les données
    logger.info("1. Chargement des données...")
    data = load_oulad_data('data/raw/open+university+learning+analytics+dataset/')
    
    logger.info("2. Préparation du dataset...")
    final_df = prepare_dataset(data)
    
    logger.info("3. Feature engineering...")
    X, y, column_names, encode_dict = engineer_features(final_df)
    
    logger.info("4. Sauvegarde des features...")
    save_features(X, y, column_names, encode_dict, output_dir='data/processed/')
    
    logger.info("5. Test de chargement...")
    X_loaded, y_loaded, cols_loaded, enc_loaded = load_features(input_dir='data/processed/')
    
    logger.info("Module build_features.py fonctionne correctement!")
except Exception as e:
    logger.error(f"\nErreur: {e}")