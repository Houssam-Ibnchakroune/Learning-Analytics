"""
Module de chargement des données vers PostgreSQL
Sauvegarde les DataFrames dans la base de données PostgreSQL
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from typing import Optional, Dict
import json
from datetime import datetime


def create_connection_string(host: str = 'localhost',
                             port: int = 5432,
                             database: str = 'oulad_db',
                             user: str = 'postgres',
                             password: str = 'postgres') -> str:
    """
    Crée une chaîne de connexion PostgreSQL.
    
    Parameters:
    -----------
    host : str
        Hôte de la base de données
    port : int
        Port PostgreSQL
    database : str
        Nom de la base de données
    user : str
        Nom d'utilisateur
    password : str
        Mot de passe
        
    Returns:
    --------
    str
        Chaîne de connexion SQLAlchemy
    """
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
        
        print(f"✅ Table '{table_name}' sauvegardée: {df.shape[0]} lignes")
        
        engine.dispose()
        
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde de '{table_name}': {e}")
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
    # Créer un DataFrame à partir de X et y
    df = pd.DataFrame(X, columns=column_names)
    df['target'] = y
    
    # Sauvegarder
    save_to_postgres(df, table_name, connection_string)


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
    # Ajouter un timestamp
    predictions['prediction_date'] = datetime.now()
    
    # Sauvegarder
    save_to_postgres(
        predictions, 
        table_name, 
        connection_string,
        if_exists='append'  # Ajouter aux prédictions existantes
    )


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
    try:
        df.to_csv(output_path, index=index, encoding='utf-8')
        print(f"✅ Backup CSV créé: {output_path}")
        print(f"   - Dimensions: {df.shape}")
        
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde CSV: {e}")
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
    try:
        engine = create_engine(connection_string)
        
        if query is None:
            query = f"SELECT * FROM {table_name}"
        
        df = pd.read_sql(query, engine)
        
        print(f"✅ Table '{table_name}' chargée: {df.shape[0]} lignes")
        
        engine.dispose()
        
        return df
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement de '{table_name}': {e}")
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
    try:
        engine = create_engine(connection_string)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version();"))
            version = result.fetchone()[0]
            print(f"✅ Connexion PostgreSQL réussie!")
            print(f"   Version: {version}")
        engine.dispose()
        return True
        
    except Exception as e:
        print(f"❌ Erreur de connexion PostgreSQL: {e}")
        return False


if __name__ == "__main__":
    # Test du module
    print("🔍 Test du module load.py\n")
    
    # Configuration de test
    conn_str = create_connection_string(
        host='localhost',
        database='oulad_db',
        user='postgres',
        password='postgres'
    )
    
    print(f"🔗 Chaîne de connexion créée")
    print(f"   (Remplacez les paramètres par vos identifiants réels)")
    
    # Test de connexion (commenté car nécessite PostgreSQL installé)
    # test_connection(conn_str)
    
    print("\n💡 Pour utiliser ce module:")
    print("   1. Installez PostgreSQL")
    print("   2. Créez une base de données: CREATE DATABASE oulad_db;")
    print("   3. Utilisez save_to_postgres() pour sauvegarder vos données")
    
    print("\n✅ Module load.py prêt à l'emploi!")