"""
Module d'extraction des données OULAD
Charge les fichiers CSV bruts et retourne un dictionnaire de DataFrames
"""

import pandas as pd
import os
from typing import Dict


def load_oulad_data(data_path: str = '../data/raw/open+university+learning+analytics+dataset/') -> Dict[str, pd.DataFrame]:
    """
    Charge tous les fichiers CSV OULAD depuis le répertoire spécifié.
    
    Parameters:
    -----------
    data_path : str
        Chemin vers le répertoire contenant les fichiers CSV OULAD
        
    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionnaire où les clés sont les noms des fichiers (sans .csv)
        et les valeurs sont les DataFrames correspondants
        
    Example:
    --------
    >>> data_dicts = load_oulad_data()
    >>> print(data_dicts.keys())
    dict_keys(['assessments', 'courses', 'studentAssessment', ...])
    """
    data_dicts = {}
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Le répertoire {data_path} n'existe pas")
    
    csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
    
    if not csv_files:
        raise ValueError(f"Aucun fichier CSV trouvé dans {data_path}")
    
    for csv_name in csv_files:
        file_key = csv_name[:-4]  # Retirer l'extension .csv
        file_path = os.path.join(data_path, csv_name)
        
        try:
            # Essayer d'abord UTF-8
            data_dicts[file_key] = pd.read_csv(file_path)
        except UnicodeDecodeError:
            # Si échec, utiliser latin-1 (ISO-8859-1)
            data_dicts[file_key] = pd.read_csv(file_path, encoding='latin-1')
            print(f"⚠️  {csv_name} chargé avec encoding latin-1")
    
    print(f"✅ {len(data_dicts)} fichiers CSV chargés")
    print(f"📁 Fichiers: {list(data_dicts.keys())}")
    
    return data_dicts


def get_dataset_info(data_dicts: Dict[str, pd.DataFrame]) -> None:
    """
    Affiche des informations sur les datasets chargés.
    
    Parameters:
    -----------
    data_dicts : Dict[str, pd.DataFrame]
        Dictionnaire de DataFrames OULAD
    """
    print("\n📊 Informations sur les datasets:")
    print("-" * 60)
    
    for name, df in data_dicts.items():
        print(f"{name:25} | Lignes: {df.shape[0]:>8,} | Colonnes: {df.shape[1]:>3}")
    
    print("-" * 60)
    total_rows = sum(df.shape[0] for df in data_dicts.values())
    print(f"{'TOTAL':25} | Lignes: {total_rows:>8,}")


if __name__ == "__main__":
    # Test du module
    print("🔍 Test du module extract.py\n")
    
    try:
        data = load_oulad_data()
        get_dataset_info(data)
        
        print("\n✅ Module extract.py fonctionne correctement!")
        
    except Exception as e:
        print(f"\n❌ Erreur: {e}")