"""
Module de transformation des données OULAD
Nettoie, merge et prépare les données pour le feature engineering
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging
logger = logging.getLogger(__name__)

def inner_merge(left_df: pd.DataFrame, 
                right_df: pd.DataFrame, 
                right_cols: list, 
                on_cols: list) -> pd.DataFrame:
    """
    Effectue un merge interne et supprime les doublons.
    
    Parameters:
    -----------
    left_df : pd.DataFrame
        DataFrame de gauche
    right_df : pd.DataFrame
        DataFrame de droite
    right_cols : list
        Colonnes à conserver du DataFrame de droite
    on_cols : list
        Colonnes sur lesquelles effectuer le merge
        
    Returns:
    --------
    pd.DataFrame
        DataFrame mergé sans doublons
    """
    right_df = right_df[right_cols]
    merged_df = left_df.merge(right_df, on=on_cols)
    return merged_df.drop_duplicates()


def create_score_df(data_dicts: Dict[str, pd.DataFrame], 
                    score_deadline: int = 90) -> pd.DataFrame:
    """
    Crée un DataFrame avec les scores moyens des étudiants jusqu'à score_deadline jours.
    
    Parameters:
    -----------
    data_dicts : Dict[str, pd.DataFrame]
        Dictionnaire contenant les DataFrames OULAD
    score_deadline : int
        Nombre de jours à inclure dans l'analyse
        
    Returns:
    --------
    pd.DataFrame
        DataFrame avec les scores moyens par étudiant
    """
    df1 = data_dicts['assessments'].copy()
    df2 = data_dicts['studentAssessment'].copy()
    
    # Convertir les colonnes en numérique
    df1['date'] = pd.to_numeric(df1['date'], errors='coerce')
    df1['weight'] = pd.to_numeric(df1['weight'], errors='coerce')
    df2['date_submitted'] = pd.to_numeric(df2['date_submitted'], errors='coerce')
    df2['score'] = pd.to_numeric(df2['score'], errors='coerce')
    df2['is_banked'] = pd.to_numeric(df2['is_banked'], errors='coerce')

    # Merge des données d'évaluation
    score_df = inner_merge(df1, df2, df2.columns, ['id_assessment'])

    # Filtrer par deadline et type d'évaluation
    score_df = score_df[score_df['date'] < score_deadline]
    score_df = score_df[score_df['assessment_type'] != 'Exam']
    score_df = score_df.dropna(subset=['score'])

    # Calculer la moyenne par étudiant
    score_df = score_df.groupby(
        ['code_module', 'code_presentation', 'id_student'], 
        as_index=False
    ).agg({
        'score': 'mean',
        'date': 'mean',
        'weight': 'mean',
        'date_submitted': 'mean',
        'is_banked': 'mean',
        'id_assessment': 'mean'
    })
    
    # Renommer et nettoyer
    score_df = score_df.rename(columns={'score': f'mean_score_day{score_deadline}'})
    score_df = score_df.drop(
        ['date', 'weight', 'date_submitted', 'is_banked', 'id_assessment'], 
        axis=1
    )
    
    return score_df


def create_click_df(data_dicts: Dict[str, pd.DataFrame], 
                    click_deadline: int = 90) -> pd.DataFrame:
    """
    Crée un DataFrame avec le nombre moyen de clics par type d'activité.
    
    Parameters:
    -----------
    data_dicts : Dict[str, pd.DataFrame]
        Dictionnaire contenant les DataFrames OULAD
    click_deadline : int
        Nombre de jours à inclure dans l'analyse
        
    Returns:
    --------
    pd.DataFrame
        DataFrame avec les clics moyens par type d'activité
    """
    clicks = data_dicts['studentVle'].copy()
    
    # Convertir les colonnes en numérique
    clicks['date'] = pd.to_numeric(clicks['date'], errors='coerce')
    clicks['sum_click'] = pd.to_numeric(clicks['sum_click'], errors='coerce')

    # Merge avec les informations VLE
    clicks = inner_merge(
        clicks,
        data_dicts['vle'],
        ['id_site', 'code_module', 'code_presentation', 'activity_type'],
        ['id_site', 'code_module', 'code_presentation']
    )

    clicks = clicks.drop('id_site', axis=1)

    # Fonction auxiliaire pour calculer les clics
    def clicks_xx(clicks_df, xx):
        temp = clicks_df[clicks_df['date'] <= xx]
        temp = temp.drop('date', axis=1)
        temp = temp.groupby(
            ['code_module', 'code_presentation', 'id_student', 'activity_type']
        ).mean()
        temp = temp.rename(columns={'sum_click': f'sum_click{xx} mean'})
        temp = temp.reset_index()
        return temp

    # Pivot pour avoir une colonne par type d'activité
    click_data = pd.pivot_table(
        data=clicks_xx(clicks, click_deadline), 
        index=['code_module', 'code_presentation', 'id_student'],
        columns='activity_type', 
        values=[f'sum_click{click_deadline} mean'],
        fill_value=0
    ).reset_index()

    # Retirer le multi-index
    click_data = pd.concat([
        click_data['code_module'],
        click_data['code_presentation'],
        click_data['id_student'], 
        click_data[f'sum_click{click_deadline} mean']
    ], axis=1)
    
    return click_data


def create_final_df(data_dicts: Dict[str, pd.DataFrame],
                    score_df: pd.DataFrame,
                    click_df: pd.DataFrame,
                    withdraw_deadline: int = 90) -> pd.DataFrame:
    """
    Merge tous les DataFrames et crée le dataset final.
    
    Parameters:
    -----------
    data_dicts : Dict[str, pd.DataFrame]
        Dictionnaire contenant les DataFrames OULAD
    score_df : pd.DataFrame
        DataFrame des scores
    click_df : pd.DataFrame
        DataFrame des clics
    withdraw_deadline : int
        Seuil de jours pour considérer les retraits
        
    Returns:
    --------
    pd.DataFrame
        DataFrame final prêt pour le feature engineering
    """
    # Merge avec les informations étudiants
    final_df = inner_merge(
        click_df,
        data_dicts['studentInfo'],
        data_dicts['studentInfo'].columns,
        ['code_module', 'code_presentation', 'id_student']
    )

    # Merger Pass et Distinction
    final_df = final_df.replace('Distinction', 'Pass')

    # Merge avec les données d'inscription
    student_reg = data_dicts['studentRegistration'].copy()
    student_reg['date_unregistration'] = pd.to_numeric(
        student_reg['date_unregistration'], errors='coerce'
    )
    student_reg['date_registration'] = pd.to_numeric(
        student_reg['date_registration'], errors='coerce'
    )
    
    final_df = inner_merge(
        final_df, 
        student_reg,
        ['code_module', 'code_presentation', 'id_student', 'date_unregistration'],
        ['code_module', 'code_presentation', 'id_student']
    )

    # Retirer les étudiants qui se sont retirés avant le seuil
    final_df = final_df[
        (final_df['final_result'] != 'Withdrawn') | 
        (final_df['date_unregistration'] > withdraw_deadline)
    ]

    final_df = final_df.reset_index()
    final_df = final_df.drop(['date_unregistration', 'index'], axis=1)

    # Merge avec les scores
    final_df = inner_merge(
        final_df,
        score_df,
        score_df.columns,
        ['code_module', 'code_presentation', 'id_student']
    )
    
    # Merger Withdrawn dans Fail
    final_df = final_df.replace('Withdrawn', 'Fail')

    return final_df


def prepare_dataset(data_dicts: Dict[str, pd.DataFrame],
                    score_deadline: int = 90,
                    click_deadline: int = 90,
                    withdraw_deadline: int = 90) -> pd.DataFrame:
    """
    Pipeline complet de préparation des données.
    
    Parameters:
    -----------
    data_dicts : Dict[str, pd.DataFrame]
        Dictionnaire contenant les DataFrames OULAD
    score_deadline : int
        Nombre de jours pour les scores
    click_deadline : int
        Nombre de jours pour les clics
    withdraw_deadline : int
        Seuil pour les retraits
        
    Returns:
    --------
    pd.DataFrame
        DataFrame final préparé
    """
    logger.info(f"Préparation du dataset...")
    logger.info(f"   - Score deadline: {score_deadline} jours")
    logger.info(f"   - Click deadline: {click_deadline} jours")
    logger.info(f"   - Withdraw deadline: {withdraw_deadline} jours")
    
    # Créer les DataFrames intermédiaires
    logger.debug("Creating intermediate DataFrames")
    score_df = create_score_df(data_dicts, score_deadline)
    logger.info(f"Score DataFrame créé: {score_df.shape}")
    
    logger.debug("Creating click DataFrame")
    click_df = create_click_df(data_dicts, click_deadline)
    logger.info(f"Click DataFrame créé: {click_df.shape}")
    
    # Créer le DataFrame final
    logger.debug("Creating final DataFrame")
    final_df = create_final_df(data_dicts, score_df, click_df, withdraw_deadline)
    logger.info(f"Final DataFrame créé: {final_df.shape}")
    
    return final_df


if __name__ == "__main__":
    # Test du module
    from extract import load_oulad_data
    
    logger.info("Test du module transform.py\n")
    
    try:
        # Charger les données
        data = load_oulad_data()
        
        # Préparer le dataset
        final_df = prepare_dataset(data)
        
        logger.info(f"\nDataset final:")
        logger.info(f"   - Dimensions: {final_df.shape}")
        logger.info(f"   - Colonnes: {list(final_df.columns)}")
        logger.info(f"\nModule transform.py fonctionne correctement!")
        
    except Exception as e:
        logger.error(f"\nErreur: {e}")