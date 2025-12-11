"""
Script de configuration de la base de données PostgreSQL pour OULAD
Crée la base de données et toutes les tables nécessaires
"""

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import sys
from pathlib import Path

# Ajouter le répertoire parent au path pour importer src
sys.path.insert(0, str(Path(__file__).parent))

from src.data.load import load_db_config


def create_database():
    """
    Crée la base de données OULAD si elle n'existe pas.
    Se connecte d'abord à postgres (base par défaut) pour créer la DB.
    """
    print("Création de la base de données...\n")
    
    # Charger la config
    config = load_db_config()
    db_name = config['database']
    
    try:
        # Connexion à la base postgres par défaut
        conn = psycopg2.connect(
            host=config['host'],
            port=config['port'],
            database='postgres',  # Base par défaut
            user=config['user'],
            password=config['password']
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Vérifier si la base existe déjà
        cursor.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (db_name,)
        )
        exists = cursor.fetchone()
        
        if exists:
            print(f"La base de données '{db_name}' existe déjà")
        else:
            # Créer la base de données
            cursor.execute(f"CREATE DATABASE {db_name}")
            print(f"Base de données '{db_name}' créée avec succès!")
        
        cursor.close()
        conn.close()
        
        return True
        
    except psycopg2.Error as e:
        print(f"Erreur lors de la création de la base: {e}")
        return False


def create_tables():
    """
    Crée toutes les tables nécessaires pour le projet OULAD.
    """
    print("\nCréation des tables...\n")
    
    config = load_db_config()
    
    try:
        # Connexion à la base OULAD
        conn = psycopg2.connect(
            host=config['host'],
            port=config['port'],
            database=config['database'],
            user=config['user'],
            password=config['password']
        )
        cursor = conn.cursor()
        
        # SQL pour créer les tables
        tables_sql = """
        -- Table des données préparées (final_df)
        CREATE TABLE IF NOT EXISTS students_prepared (
            id SERIAL PRIMARY KEY,
            code_module VARCHAR(10),
            code_presentation VARCHAR(10),
            id_student INTEGER,
            gender VARCHAR(1),
            region VARCHAR(50),
            highest_education VARCHAR(50),
            age_band VARCHAR(20),
            disability VARCHAR(1),
            final_result VARCHAR(20),
            num_of_prev_attempts INTEGER,
            studied_credits INTEGER,
            mean_score_day90 FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Table des features ML (X, y)
        CREATE TABLE IF NOT EXISTS ml_features (
            id SERIAL PRIMARY KEY,
            code_module VARCHAR(10),
            code_presentation VARCHAR(10),
            gender INTEGER,
            region INTEGER,
            highest_education INTEGER,
            age_band INTEGER,
            disability INTEGER,
            num_of_prev_attempts INTEGER,
            studied_credits INTEGER,
            mean_score_day90 FLOAT,
            target INTEGER,  -- 0=Pass, 1=Fail
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Table des prédictions du modèle
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            id_student INTEGER,
            code_module VARCHAR(10),
            code_presentation VARCHAR(10),
            prediction INTEGER,  -- 0=Pass, 1=Fail
            probability_fail FLOAT,
            probability_pass FLOAT,
            risk_score FLOAT,  -- 0-1
            risk_level VARCHAR(20),  -- Faible/Moyen/Élevé
            prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            model_version VARCHAR(50)
        );
        
        -- Table des métadonnées des modèles
        CREATE TABLE IF NOT EXISTS model_metadata (
            id SERIAL PRIMARY KEY,
            model_id VARCHAR(50) UNIQUE,
            model_type VARCHAR(50),
            n_estimators INTEGER,
            max_features VARCHAR(20),
            min_samples_split INTEGER,
            n_features INTEGER,
            feature_names TEXT[],  -- Tableau de noms de features
            training_date TIMESTAMP,
            training_samples INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Table des métriques d'évaluation
        CREATE TABLE IF NOT EXISTS model_metrics (
            id SERIAL PRIMARY KEY,
            model_id VARCHAR(50),
            metric_name VARCHAR(50),
            metric_value FLOAT,
            dataset_type VARCHAR(20),  -- train/test/validation
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (model_id) REFERENCES model_metadata(model_id)
        );
        
        -- Table des interventions (pour le suivi des étudiants à risque)
        CREATE TABLE IF NOT EXISTS student_interventions (
            id SERIAL PRIMARY KEY,
            id_student INTEGER,
            code_module VARCHAR(10),
            code_presentation VARCHAR(10),
            risk_level VARCHAR(20),
            intervention_type VARCHAR(50),
            intervention_date DATE,
            status VARCHAR(20),  -- planned/completed/cancelled
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        # Exécuter le SQL
        cursor.execute(tables_sql)
        conn.commit()
        
        print("Tables créées avec succès:")
        print("   - students_prepared")
        print("   - ml_features")
        print("   - predictions")
        print("   - model_metadata")
        print("   - model_metrics")
        print("   - student_interventions")
        
        cursor.close()
        conn.close()
        
        return True
        
    except psycopg2.Error as e:
        print(f"Erreur lors de la création des tables: {e}")
        return False


def create_indexes():
    """
    Crée des index pour améliorer les performances des requêtes.
    """
    print("\nCréation des index...\n")
    
    config = load_db_config()
    
    try:
        conn = psycopg2.connect(
            host=config['host'],
            port=config['port'],
            database=config['database'],
            user=config['user'],
            password=config['password']
        )
        cursor = conn.cursor()
        
        indexes_sql = """
        -- Index pour les recherches fréquentes
        CREATE INDEX IF NOT EXISTS idx_students_id ON students_prepared(id_student);
        CREATE INDEX IF NOT EXISTS idx_students_module ON students_prepared(code_module, code_presentation);
        CREATE INDEX IF NOT EXISTS idx_students_result ON students_prepared(final_result);
        
        CREATE INDEX IF NOT EXISTS idx_predictions_student ON predictions(id_student);
        CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(prediction_date);
        CREATE INDEX IF NOT EXISTS idx_predictions_risk ON predictions(risk_level);
        
        CREATE INDEX IF NOT EXISTS idx_model_id ON model_metadata(model_id);
        CREATE INDEX IF NOT EXISTS idx_model_date ON model_metadata(training_date);
        
        CREATE INDEX IF NOT EXISTS idx_interventions_student ON student_interventions(id_student);
        CREATE INDEX IF NOT EXISTS idx_interventions_status ON student_interventions(status);
        """
        
        cursor.execute(indexes_sql)
        conn.commit()
        
        print("Index créés avec succès")
        
        cursor.close()
        conn.close()
        
        return True
        
    except psycopg2.Error as e:
        print(f"Erreur lors de la création des index: {e}")
        return False


def verify_setup():
    """
    Vérifie que la base et les tables ont été créées correctement.
    """
    print("\nVérification de la configuration...\n")
    
    config = load_db_config()
    
    try:
        conn = psycopg2.connect(
            host=config['host'],
            port=config['port'],
            database=config['database'],
            user=config['user'],
            password=config['password']
        )
        cursor = conn.cursor()
        
        # Lister les tables
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        
        tables = cursor.fetchall()
        
        print(f"Tables trouvées dans '{config['database']}':")
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
            count = cursor.fetchone()[0]
            print(f"   - {table[0]:30} ({count} lignes)")
        
        cursor.close()
        conn.close()
        
        return True
        
    except psycopg2.Error as e:
        print(f"Erreur lors de la vérification: {e}")
        return False


def main():
    """
    Fonction principale : exécute toutes les étapes de configuration.
    """
    print("=" * 60)
    print("  CONFIGURATION DE LA BASE DE DONNÉES OULAD")
    print("=" * 60)
    
    # Étape 1 : Créer la base de données
    if not create_database():
        print("\nÉchec de la création de la base de données")
        return False
    
    # Étape 2 : Créer les tables
    if not create_tables():
        print("\nÉchec de la création des tables")
        return False
    
    # Étape 3 : Créer les index
    if not create_indexes():
        print("\nÉchec de la création des index")
        return False
    
    # Étape 4 : Vérifier
    if not verify_setup():
        print("\nÉchec de la vérification")
        return False
    
    print("\n" + "=" * 60)
    print("  CONFIGURATION TERMINÉE AVEC SUCCÈS!")
    print("=" * 60)
    print("\nProchaines étapes:")
    print("   1. Exécutez le pipeline ETL: python pipeline.py")
    print("   2. Entraînez le modèle ML")
    print("   3. Connectez Power BI à PostgreSQL")
    
    return True


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nConfiguration interrompue par l'utilisateur")
    except Exception as e:
        print(f"\nErreur inattendue: {e}")
        import traceback
        traceback.print_exc()
