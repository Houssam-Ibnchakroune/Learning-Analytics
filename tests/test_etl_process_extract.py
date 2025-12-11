from src.data.extract import load_oulad_data, get_dataset_info
from utils.login_setup import setup_logging
setup_logging()
import logging
logger = logging.getLogger(__name__)
logger.info("Test du module extract.py\n")
    
try:
        data = load_oulad_data('data/raw/open+university+learning+analytics+dataset/')
        get_dataset_info(data)
        
        logger.info("\nModule extract.py fonctionne correctement!")
        
except Exception as e:
        logger.error(f"\nErreur: {e}")