"""
Module pour l'évaluation des performances des modèles.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score
)


def evaluate_classification(y_true, y_pred, class_names=None):
    """
    Calcule les métriques de classification.
    
    Args:
        y_true: Vrais labels
        y_pred: Prédictions
        class_names: Noms des classes (optionnel)
        
    Returns:
        metrics: Dictionnaire avec accuracy, precision, recall, F1
    """
    if class_names is None:
        class_names = ['Pass', 'Fail']
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted')
    }
    
    # Affichage
    print("=" * 50)
    print("MÉTRIQUES D'ÉVALUATION")
    print("=" * 50)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print("=" * 50)
    
    # Rapport de classification détaillé
    print("\nRapport de classification détaillé :")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, save_path=None, class_names=None):
    """
    Génère et affiche la matrice de confusion.
    
    Args:
        y_true: Vrais labels
        y_pred: Prédictions
        save_path: Chemin de sauvegarde (optionnel)
        class_names: Noms des classes (optionnel)
    """
    if class_names is None:
        class_names = ['Pass', 'Fail']
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matrice de Confusion', fontsize=16, fontweight='bold')
    plt.ylabel('Vraie Classe', fontsize=12)
    plt.xlabel('Classe Prédite', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Matrice de confusion sauvegardée : {save_path}")
    
    plt.show()
    
    return cm


def plot_roc_curve(y_true, y_proba, save_path=None):
    """
    Génère et affiche la courbe ROC.
    
    Args:
        y_true: Vrais labels
        y_proba: Probabilités prédites (pour classe positive)
        save_path: Chemin de sauvegarde (optionnel)
        
    Returns:
        auc_score: Score AUC
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc_score = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Courbe ROC (Receiver Operating Characteristic)', 
              fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Courbe ROC sauvegardée : {save_path}")
    
    plt.show()
    
    return auc_score


def get_feature_importance(model, feature_names):
    """
    Extrait l'importance des features du modèle.
    
    Args:
        model: Modèle Random Forest ou similaire
        feature_names: Liste des noms de features
        
    Returns:
        importance_df: DataFrame avec features triées par importance
    """
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Le modèle ne supporte pas feature_importances_")
    
    importances = model.feature_importances_
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("=" * 50)
    print("🔝 TOP 10 FEATURES LES PLUS IMPORTANTES")
    print("=" * 50)
    print(importance_df.head(10).to_string(index=False))
    print("=" * 50)
    
    return importance_df


def plot_feature_importance(importance_df, top_n=15, save_path=None):
    """
    Visualise l'importance des features.
    
    Args:
        importance_df: DataFrame avec colonnes 'feature' et 'importance'
        top_n: Nombre de top features à afficher
        save_path: Chemin de sauvegarde (optionnel)
    """
    top_features = importance_df.head(top_n)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance', fontsize=12)
    plt.title(f'Top {top_n} Features les plus importantes', 
              fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance sauvegardée : {save_path}")
    
    plt.show()


def evaluate_model_complete(model, X_test, y_test, y_proba=None, 
                           feature_names=None, save_dir=None):
    """
    Évaluation complète d'un modèle (métriques + visualisations).
    
    Args:
        model: Modèle entraîné
        X_test: Features de test
        y_test: Vrais labels
        y_proba: Probabilités (optionnel, calculées si non fournies)
        feature_names: Noms des features (optionnel)
        save_dir: Répertoire pour sauvegarder graphiques (optionnel)
        
    Returns:
        results: Dictionnaire avec toutes les métriques
    """
    # Prédictions
    y_pred = model.predict(X_test)
    
    if y_proba is None and hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
    
    # Métriques
    metrics = evaluate_classification(y_test, y_pred)
    
    # Matrice de confusion
    cm_path = f"{save_dir}/confusion_matrix.png" if save_dir else None
    plot_confusion_matrix(y_test, y_pred, save_path=cm_path)
    
    # Courbe ROC
    if y_proba is not None:
        roc_path = f"{save_dir}/roc_curve.png" if save_dir else None
        auc_score = plot_roc_curve(y_test, y_proba, save_path=roc_path)
        metrics['auc'] = auc_score
    
    # Feature importance
    if feature_names is not None and hasattr(model, 'feature_importances_'):
        importance_df = get_feature_importance(model, feature_names)
        fi_path = f"{save_dir}/feature_importance.png" if save_dir else None
        plot_feature_importance(importance_df, save_path=fi_path)
        
        # Sauvegarder CSV
        if save_dir:
            csv_path = f"{save_dir}/feature_importance.csv"
            importance_df.to_csv(csv_path, index=False)
            print(f"Feature importance CSV sauvegardée : {csv_path}")
    
    return metrics


# Exemple d'utilisation
if __name__ == "__main__":
    import sys
    sys.path.append('..')
    
    # Charger données et modèle
    # model = joblib.load('../models/random_forest_day180.pkl')
    # X_test = np.load('../data/processed/X_test.npy')
    # y_test = np.load('../data/processed/y_test.npy')
    
    # Évaluation complète
    # results = evaluate_model_complete(
    #     model, X_test, y_test,
    #     feature_names=['feature1', 'feature2', ...],
    #     save_dir='../reports/figures'
    # )
    
    print("Module evaluate.py prêt à l'emploi !")
