"""
Module de chargement des données vers PostgreSQL
Sauvegarde les DataFrames dans la base de données PostgreSQL
"""
import logging
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from typing import Optional, Dict
from config.config import HOST,USER ,PW, DB, PORT
import os
from datetime import datetime

logger = logging.getLogger(__name__)

def load_db_config() -> Dict[str, str]:
    """
    Charge la configuration de la base de données depuis les variables d'environnement.
    
    Essaie d'abord de charger depuis un fichier .env avec python-dotenv,
    sinon utilise les variables d'environnement système.
    
    Returns:
    --------
    Dict[str, str]
        Dictionnaire contenant les paramètres de connexion
        
    Raises:
    -------
    ValueError
        Si les variables d'environnement requises ne sont pas définies
    """
    try:
        config = {
            'host': HOST,
            'port': PORT,
            'database': DB,
            'user': USER,
            'password': PW
        }
    except KeyError as e:
        logger.error(f"Variable d'environnement manquante: {e}")
        raise ValueError(f"Variable d'environnement manquante: {e}")
        
    return config


def create_connection_string(host: Optional[str] = None,
                             port: Optional[int] = None,
                             database: Optional[str] = None,
                             user: Optional[str] = None,
                             password: Optional[str] = None,
                             use_env: bool = True) -> str:
    """
    Crée une chaîne de connexion PostgreSQL.
    
    Si use_env=True (par défaut), charge la config depuis les variables d'environnement.
    Sinon, utilise les paramètres fournis.
    
    Parameters:
    -----------
    host : Optional[str]
        Hôte de la base de données
    port : Optional[int]
        Port PostgreSQL
    database : Optional[str]
        Nom de la base de données
    user : Optional[str]
        Nom d'utilisateur
    password : Optional[str]
        Mot de passe
    use_env : bool
        Utiliser les variables d'environnement (recommandé)
        
    Returns:
    --------
    str
        Chaîne de connexion SQLAlchemy
        
    Example:
    --------
    >>> # Recommandé: utiliser .env
    >>> conn_str = create_connection_string()
    
    >>> # Alternative: spécifier manuellement (non recommandé pour production)
    >>> conn_str = create_connection_string(
    ...     host='localhost',
    ...     user='postgres',
    ...     password='secret',
    ...     use_env=False
    ... )
    """
    logger.debug("Creating PostgreSQL connection string")
    if use_env:
        config = load_db_config()
        host = config['host']
        port = config['port']
        database = config['database']
        user = config['user']
        password = config['password']
    else:
        # Valeurs par défaut si non spécifiées
        host = host or 'localhost'
        port = port or 5432
        database = database or 'oulad_db'
        user = user or 'postgres'
        password = password or ''
        
        if not password:
            logger.error("Mot de passe requis si use_env=False")
            raise ValueError("Mot de passe requis!")
    
    logger.info("PostgreSQL connection string created")
    
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


def save_to_postgres(df: pd.DataFrame,
                     table_name: str,
                     connection_string: str,
                     if_exists: str = 'replace',
                     index: bool = False) -> None:
    """
    Sauvegarde un DataFrame dans PostgreSQL.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame à sauvegarder
    table_name : str
        Nom de la table dans PostgreSQL
    connection_string : str
        Chaîne de connexion PostgreSQL
    if_exists : str
        Action si la table existe: 'fail', 'replace', 'append'
    index : bool
        Inclure l'index dans la table
        
    Example:
    --------
    >>> conn_str = create_connection_string(
    ...     host='localhost',
    ...     database='oulad_db',
    ...     user='postgres',
    ...     password='mypassword'
    ... )
    >>> save_to_postgres(df, 'students', conn_str)
    """
    logger.debug(f"Saving DataFrame to PostgreSQL table '{table_name}'")
    try:
        engine = create_engine(connection_string)
        
        # Sauvegarder le DataFrame
        df.to_sql(
            name=table_name,
            con=engine,
            if_exists=if_exists,
            index=index,
            method='multi',
            chunksize=1000
        )
        
        logger.info(f"Table '{table_name}' sauvegardée: {df.shape[0]} lignes")
        
        engine.dispose()
        
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde de '{table_name}': {e}")
        raise


def save_features_to_postgres(X: np.ndarray,
                              y: np.ndarray,
                              column_names: list,
                              connection_string: str,
                              table_name: str = 'ml_features') -> None:
    """
    Sauvegarde les features (X, y) dans PostgreSQL.
    
    Parameters:
    -----------
    X : np.ndarray
        Matrice des features
    y : np.ndarray
        Vecteur des labels
    column_names : list
        Noms des colonnes
    connection_string : str
        Chaîne de connexion PostgreSQL
    table_name : str
        Nom de la table
    """
    logger.debug(f"Saving features to PostgreSQL table '{table_name}'")
    # Créer un DataFrame à partir de X et y
    df = pd.DataFrame(X, columns=column_names)
    df['target'] = y
    
    # Sauvegarder
    save_to_postgres(df, table_name, connection_string)
    logger.info(f"Features sauvegardées dans la table '{table_name}'")


def save_predictions_to_postgres(predictions: pd.DataFrame,
                                 connection_string: str,
                                 table_name: str = 'predictions') -> None:
    """
    Sauvegarde les prédictions du modèle dans PostgreSQL.
    
    Parameters:
    -----------
    predictions : pd.DataFrame
        DataFrame contenant les prédictions
        Colonnes attendues: id_student, prediction, probability, risk_score, etc.
    connection_string : str
        Chaîne de connexion PostgreSQL
    table_name : str
        Nom de la table
    """
    logger.debug(f"Saving predictions to PostgreSQL table '{table_name}'")
    # Ajouter un timestamp
    predictions['prediction_date'] = datetime.now()
    
    # Sauvegarder
    save_to_postgres(
        predictions, 
        table_name, 
        connection_string,
        if_exists='append'  # Ajouter aux prédictions existantes
    )
    logger.info(f"Prédictions sauvegardées dans la table '{table_name}'")


def save_model_metadata_to_postgres(metadata: Dict,
                                    connection_string: str,
                                    table_name: str = 'model_metadata') -> None:
    """
    Sauvegarde les métadonnées du modèle dans PostgreSQL.
    
    Parameters:
    -----------
    metadata : Dict
        Dictionnaire contenant les métadonnées du modèle
    connection_string : str
        Chaîne de connexion PostgreSQL
    table_name : str
        Nom de la table
    """
    logger.debug(f"Saving model metadata to PostgreSQL table '{table_name}'")
    # Convertir le dictionnaire en DataFrame
    df = pd.DataFrame([metadata])
    
    # Ajouter un ID unique
    df['model_id'] = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Sauvegarder
    save_to_postgres(
        df, 
        table_name, 
        connection_string,
        if_exists='append'
    )
    logger.info(f"Métadonnées du modèle sauvegardées dans la table '{table_name}'")


def save_csv_backup(df: pd.DataFrame,
                    output_path: str,
                    index: bool = False) -> None:
    """
    Sauvegarde un DataFrame en CSV (backup).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame à sauvegarder
    output_path : str
        Chemin du fichier CSV
    index : bool
        Inclure l'index
    """
    logger.debug(f"Saving DataFrame backup to CSV at '{output_path}'")
    try:
        df.to_csv(output_path, index=index, encoding='utf-8')
        logger.info(f"Backup CSV créé: {output_path}")
        logger.info(f"   - Dimensions: {df.shape}")
        
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde CSV: {e}")
        raise


def load_from_postgres(table_name: str,
                      connection_string: str,
                      query: Optional[str] = None) -> pd.DataFrame:
    """
    Charge des données depuis PostgreSQL.
    
    Parameters:
    -----------
    table_name : str
        Nom de la table à charger
    connection_string : str
        Chaîne de connexion PostgreSQL
    query : Optional[str]
        Requête SQL personnalisée (sinon SELECT * FROM table)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame chargé depuis PostgreSQL
    """
    logger.debug(f"Loading data from PostgreSQL table '{table_name}'")
    try:
        engine = create_engine(connection_string)
        
        if query is None:
            query = f"SELECT * FROM {table_name}"
        
        df = pd.read_sql(query, engine)
        
        logger.info(f"Table '{table_name}' chargée: {df.shape[0]} lignes")
        
        engine.dispose()
        logger.info(f"DataFrame chargé depuis la table '{table_name}'")
        
        return df
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement de '{table_name}': {e}")
        raise


def test_connection(connection_string: str) -> bool:
    """
    Teste la connexion à PostgreSQL.
    
    Parameters:
    -----------
    connection_string : str
        Chaîne de connexion PostgreSQL
        
    Returns:
    --------
    bool
        True si la connexion réussit
    """
    logger.debug("Testing PostgreSQL connection")
    try:
        engine = create_engine(connection_string)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version();"))
            version = result.fetchone()[0]
            logger.info(f"Connexion PostgreSQL réussie!")
            logger.info(f"   Version: {version}")
        engine.dispose()
        return True
        
    except Exception as e:
        logger.error(f"Erreur de connexion PostgreSQL: {e}")
        return False


if __name__ == "__main__":
    # Test du module
    logger.info("Test du module load.py\n")
    
    logger.info("Configuration recommandée:")
    logger.info("   1. Installez python-dotenv: pip install python-dotenv")
    logger.info("   2. Copiez .env.example vers .env")
    logger.info("   3. Remplissez .env avec vos identifiants PostgreSQL")
    logger.info("   4. Créez la base de données: CREATE DATABASE oulad_db;\n")
    
    try:
        # Test avec .env
        logger.info("Test de chargement depuis .env...")
        conn_str = create_connection_string()
        logger.info("Chaîne de connexion créée avec succès")
        
        # Test de connexion (décommentez si PostgreSQL est installé)
        test_connection(conn_str)
        
    except ValueError as e:
        logger.error(f"\n  {e}")
        logger.error("\nExemple de fichier .env:")
        logger.error("   DB_HOST=localhost")
        logger.error("   DB_PORT=5432")
        logger.error("   DB_NAME=oulad_db")
        logger.error("   DB_USER=postgres")
        logger.error("   DB_PASSWORD=votre_mot_de_passe")
    
    except Exception as e:
        logger.error(f"\nErreur: {e}")
    
    logger.info("\nModule load.py prêt à l'emploi!")