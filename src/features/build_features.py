"""
Module de feature engineering pour OULAD
Encode les variables catégorielles et gère la colinéarité
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from collections import defaultdict
from typing import Tuple, Dict


def create_Xy(final_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.Index, Dict]:
    """
    Transforme le DataFrame final en matrices X et y pour le ML.
    
    - Encode les variables catégorielles avec LabelEncoder
    - Supprime les colonnes non pertinentes
    - Retourne X (features), y (target), noms des colonnes et dictionnaire d'encodage
    
    Parameters:
    -----------
    final_df : pd.DataFrame
        DataFrame préparé contenant les données étudiants
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, pd.Index, Dict]
        - X : matrice des features (numpy array)
        - y : vecteur des labels (numpy array)
        - column_names : noms des colonnes de X
        - encode_dict : dictionnaire des mappings d'encodage
        
    Example:
    --------
    >>> X, y, column_names, encode_dict = create_Xy(final_df)
    >>> print(f"X shape: {X.shape}, y shape: {y.shape}")
    X shape: (32593, 18), y shape: (32593,)
    """
    # Créer une copie pour ne pas modifier l'original
    df = final_df.copy()
    
    # Séparer features et target
    X = df.drop(['final_result', 'id_student', 'imd_band'], axis=1)
    column_names = X.columns
    y = df['final_result']

    # Encoder les variables catégorielles
    le = LabelEncoder()
    encode_dict = {}
    
    cat_features = [
        'code_module', 
        'code_presentation',
        'gender', 
        'region',
        'highest_education',  
        'age_band',
        'disability'
    ]

    for cat_feature in cat_features: 
        X[cat_feature] = le.fit_transform(X[cat_feature])
        encode_dict[cat_feature] = le.classes_

    # Encoder la target
    y = le.fit_transform(y)
    encode_dict['final_result'] = le.classes_

    # Convertir en numpy array
    X = X.to_numpy()
    
    return X, y, column_names, encode_dict


def handle_collinearity(X: np.ndarray, 
                        column_names: pd.Index,
                        threshold: float = 1.0) -> Tuple[np.ndarray, pd.Index]:
    """
    Gère la colinéarité entre les features en utilisant clustering hiérarchique.
    
    Pour chaque cluster de features corrélées, garde seulement la première feature.
    Utilise la corrélation de Spearman et le clustering de Ward.
    
    Parameters:
    -----------
    X : np.ndarray
        Matrice des features
    column_names : pd.Index
        Noms des colonnes
    threshold : float
        Seuil de distance pour le clustering (défaut: 1.0)
        
    Returns:
    --------
    Tuple[np.ndarray, pd.Index]
        - X filtré (features non colinéaires)
        - column_names filtré
        
    Example:
    --------
    >>> X_filtered, cols_filtered = handle_collinearity(X, column_names)
    >>> print(f"Features avant: {X.shape[1]}, après: {X_filtered.shape[1]}")
    Features avant: 18, après: 15
    """
    print(f"🔧 Gestion de la colinéarité...")
    print(f"   - Features initiales: {X.shape[1]}")
    
    # Calculer la corrélation de Spearman
    corr = spearmanr(X).correlation
    
    # Clustering hiérarchique
    corr_linkage = hierarchy.ward(corr)
    cluster_ids = hierarchy.fcluster(corr_linkage, threshold, criterion='distance')
    
    # Grouper les features par cluster
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)
    
    # Garder seulement la première feature de chaque cluster
    selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
    
    # Filtrer X et column_names
    X_filtered = X[:, selected_features]
    column_names_filtered = column_names[selected_features]
    
    print(f"   - Features après filtrage: {X_filtered.shape[1]}")
    print(f"   - Features supprimées: {X.shape[1] - X_filtered.shape[1]}")
    
    return X_filtered, column_names_filtered


def engineer_features(final_df: pd.DataFrame,
                      handle_collinear: bool = True) -> Tuple[np.ndarray, np.ndarray, list, Dict]:
    """
    Pipeline complet de feature engineering.
    
    Parameters:
    -----------
    final_df : pd.DataFrame
        DataFrame préparé
    handle_collinear : bool
        Appliquer le filtrage de colinéarité
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, list, Dict]
        - X : matrice des features
        - y : vecteur des labels
        - column_names : noms des features (liste)
        - encode_dict : dictionnaire d'encodage
    """
    print(f"🔧 Feature Engineering...")
    
    # Créer X et y
    X, y, column_names, encode_dict = create_Xy(final_df)
    print(f"✅ Features créées: X{X.shape}, y{y.shape}")
    
    # Gérer la colinéarité si demandé
    if handle_collinear:
        X, column_names = handle_collinearity(X, column_names)
    
    # Convertir column_names en liste
    column_names = column_names.tolist()
    
    print(f"✅ Feature engineering terminé!")
    print(f"   - Features finales: {len(column_names)}")
    print(f"   - Exemples: {column_names[:5]}")
    
    return X, y, column_names, encode_dict


def save_features(X: np.ndarray,
                 y: np.ndarray,
                 column_names: list,
                 encode_dict: Dict,
                 output_dir: str = '../data/processed/') -> None:
    """
    Sauvegarde les features et métadonnées.
    
    Parameters:
    -----------
    X : np.ndarray
        Matrice des features
    y : np.ndarray
        Vecteur des labels
    column_names : list
        Noms des features
    encode_dict : Dict
        Dictionnaire d'encodage
    output_dir : str
        Répertoire de sortie
    """
    import os
    import json
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Sauvegarder X et y
    np.save(os.path.join(output_dir, 'X_features.npy'), X)
    np.save(os.path.join(output_dir, 'y_labels.npy'), y)
    
    # Convertir les numpy arrays en listes pour JSON
    encode_dict_serializable = {
        k: v.tolist() if hasattr(v, 'tolist') else v 
        for k, v in encode_dict.items()
    }
    
    # Sauvegarder les métadonnées
    metadata = {
        'column_names': column_names,
        'encode_dict': encode_dict_serializable,
        'n_samples': int(X.shape[0]),
        'n_features': int(X.shape[1])
    }
    
    with open(os.path.join(output_dir, 'features_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Features sauvegardées dans {output_dir}")
    print(f"   - X_features.npy: {X.shape}")
    print(f"   - y_labels.npy: {y.shape}")
    print(f"   - features_metadata.json")


def load_features(input_dir: str = '../data/processed/') -> Tuple[np.ndarray, np.ndarray, list, Dict]:
    """
    Charge les features depuis les fichiers sauvegardés.
    
    Parameters:
    -----------
    input_dir : str
        Répertoire contenant les fichiers
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, list, Dict]
        X, y, column_names, encode_dict
    """
    import os
    import json
    
    X = np.load(os.path.join(input_dir, 'X_features.npy'))
    y = np.load(os.path.join(input_dir, 'y_labels.npy'))
    
    with open(os.path.join(input_dir, 'features_metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    column_names = metadata['column_names']
    encode_dict = metadata['encode_dict']
    
    print(f"✅ Features chargées depuis {input_dir}")
    print(f"   - X: {X.shape}")
    print(f"   - y: {y.shape}")
    
    return X, y, column_names, encode_dict


if __name__ == "__main__":
    # Test du module
    from src.data.extract import load_oulad_data
    from src.data.transform import prepare_dataset
    
    print("🔍 Test du module build_features.py\n")
    
    try:
        # Charger et préparer les données
        print("1. Chargement des données...")
        data = load_oulad_data()
        
        print("\n2. Préparation du dataset...")
        final_df = prepare_dataset(data)
        
        print("\n3. Feature engineering...")
        X, y, column_names, encode_dict = engineer_features(final_df)
        
        print("\n4. Sauvegarde des features...")
        save_features(X, y, column_names, encode_dict)
        
        print("\n5. Test de chargement...")
        X_loaded, y_loaded, cols_loaded, enc_loaded = load_features()
        
        print("\n✅ Module build_features.py fonctionne correctement!")
        
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
