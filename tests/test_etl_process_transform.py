from src.data.extract import load_oulad_data
from src.data.transform import prepare_dataset
from utils.login_setup import setup_logging
setup_logging()
import logging
logger = logging.getLogger(__name__)   
logger.info("Test du module transform.py\n")

try:
    # Charger les données
    data = load_oulad_data('data/raw/open+university+learning+analytics+dataset/')
    
    # Préparer le dataset
    final_df = prepare_dataset(data)
    
    logger.info(f"\nDataset final:")
    logger.info(f"   - Dimensions: {final_df.shape}")
    logger.info(f"   - Colonnes: {list(final_df.columns)}")
    logger.info(f" \n\nhead : {final_df.head()}")
    logger.info(f"\nModule transform.py fonctionne correctement!")
    
except Exception as e:
    logger.error(f"\nErreur: {e}")