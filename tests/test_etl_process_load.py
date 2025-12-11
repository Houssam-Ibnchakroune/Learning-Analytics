from src.data.load import save_features_to_postgres, save_model_metadata_to_postgres
from src.data.load import create_connection_string
from src.data.load import test_connection, load_from_postgres
from utils.login_setup import setup_logging
import numpy as np  # ← AJOUTER CET IMPORT
import json
import logging
setup_logging()
import logging
logger = logging.getLogger(__name__)  
logger.info("Test du module etl_process_load.py\n")

try:
    connection_string = create_connection_string()
    X = np.load('data/processed/X_features.npy')
    y = np.load('data/processed/y_labels.npy')
    
    # Charger les noms de colonnes depuis features_metadata.json
    with open('data/processed/features_metadata.json', 'r') as f:
        metadata = json.load(f)
    column_names = metadata['column_names']
    
    logger.info(f"Chargement des features: X shape={X.shape}, y shape={y.shape}")
    logger.info(f"Nombre de colonnes: {len(column_names)}")
    
    save_features_to_postgres(X=X, y=y, column_names=column_names, connection_string=connection_string)
    with open('models/model_metadata_90_day.json', 'r') as f:
        metadata = json.load(f)
    save_model_metadata_to_postgres(metadata=metadata, connection_string=connection_string)
    
    model_metadata = load_from_postgres(connection_string=connection_string, table_name='model_metadata')
    ml_features = load_from_postgres(connection_string=connection_string, table_name='ml_features')
    print(model_metadata.head())
    print(ml_features.head())
except Exception as e:
    logger.error(f"\nErreur: {e}")