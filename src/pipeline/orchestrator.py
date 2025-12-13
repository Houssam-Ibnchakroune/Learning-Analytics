"""
Pipeline Orchestrator - Automatisation complète du workflow ML
"""

import argparse
from importlib.metadata import metadata
import logging
import time
from datetime import datetime
from pathlib import Path
import json
import sys
import numpy as np

# Imports des modules du projet
from src.data.extract import load_oulad_data
from src.data.transform import prepare_dataset
from src.data.load import (
    create_connection_string,
    save_features_to_postgres,
    save_model_metadata_to_postgres,
    save_to_postgres
)
from src.features.build_features import (
    engineer_features,
    save_features,
    load_features
)
from src.models.train import (
    train_with_cross_validation,
    save_model,
    save_model_metadata
)
from src.models.evaluate import evaluate_model_complete
from src.models.predict import predict_batch

from utils.login_setup import setup_logging

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    Orchestrateur principal du pipeline ML.
    
    Modes disponibles:
    - dev: Développement avec échantillons réduits
    - prod: Production avec données complètes
    """
    
    def __init__(self, mode='dev', config=None, cutoff_days=180):
        """
        Initialise l'orchestrateur.
        
        Args:
            mode: 'dev' ou 'prod'
            config: Dictionnaire de configuration (optionnel)
            cutoff_days: Nombre de jours pour le cutoff_date (défaut: 180)
        """
        self.mode = mode
        self.cutoff_days = cutoff_days
        self.config = config or self._get_default_config()
        self.start_time = None
        self.results = {}
        
        # Chemins selon le mode
        self.paths = self._setup_paths()
        
        logger.info("="*70)
        logger.info(f"PIPELINE ORCHESTRATOR - MODE: {mode.upper()} - CUTOFF: {cutoff_days} jours")
        logger.info("="*70)
    
    def _get_default_config(self):
        """Configuration par défaut du pipeline."""
        return {
            'dev': {
                'sample_size': 5000,
                'n_splits_cv': 3,
                'model_params': {
                    'n_estimators': 100,
                    'max_features': 'sqrt',
                    'min_samples_split': 10,
                    'random_state': 42
                }
            },
            'prod': {
                'sample_size': None,  # Toutes les données
                'n_splits_cv': 5,
                'model_params': {
                    'n_estimators': 200,
                    'max_features': 'sqrt',
                    'min_samples_split': 10,
                    'random_state': 42
                }
            }
        }
    
    def _setup_paths(self):
        """Configure les chemins selon le mode."""
        suffix = '_dev' if self.mode == 'dev' else '_prod'
        
        return {
            'raw_data': 'data/raw/open+university+learning+analytics+dataset/',
            'processed_data': 'data/processed',
            'features': f'data/processed/X_features.npy',
            'labels': f'data/processed/y_labels.npy',
            'metadata': f'data/processed/features_metadata.json',
            'model': f'models/random_forest_day{self.cutoff_days}{suffix}.pkl',
            'model_metadata': f'models/model_metadata_day{self.cutoff_days}{suffix}.json',
            'predictions': f'reports/predictions{suffix}.csv',
            'evaluation': f'reports/figures{suffix}',
            'logs': f'logs/pipeline{suffix}.log'
        }
    
    def run(self, steps=None):
        """
        Exécute le pipeline complet ou des étapes spécifiques.
        
        Args:
            steps: Liste des étapes à exécuter (None = toutes)
                   Options: ['extract', 'transform', 'load', 'features', 
                            'train', 'evaluate', 'predict']
        """
        self.start_time = time.time()
        
        all_steps = ['extract', 'transform', 'load', 'features', 
                     'train', 'evaluate', 'predict']
        steps_to_run = steps or all_steps
        
        logger.info(f"Étapes à exécuter: {', '.join(steps_to_run)}")
        logger.info("")
        
        try:
            if 'extract' in steps_to_run:
                self._step_extract()
            
            if 'transform' in steps_to_run:
                self._step_transform()
            
            if 'features' in steps_to_run:
                self._step_features()
            
            if 'train' in steps_to_run:
                self._step_train()
            
            if 'evaluate' in steps_to_run:
                self._step_evaluate()
            
            if 'predict' in steps_to_run:
                self._step_predict()
            
            if 'load' in steps_to_run:
                self._step_load()

            self._print_summary()
            
        except Exception as e:
            logger.error(f"ERREUR CRITIQUE: {str(e)}", exc_info=True)
            raise
    
    def _step_extract(self):
        """Étape 1: Extraction des données."""
        logger.info("ÉTAPE 1/7: EXTRACTION DES DONNÉES")
        logger.info("-" * 70)
        
        start = time.time()
        
        # Extraction
        raw_data = load_oulad_data(self.paths['raw_data'])
        self.raw_data = raw_data  # Sauvegarder pour étapes suivantes
        
        elapsed = time.time() - start
        self.results['extract'] = {
            'duration': elapsed,
            'files_count': len(raw_data),
            'status': 'success'
        }
        
        logger.info(f"Extraction terminée en {elapsed:.2f}s")
        logger.info("")
    
    def _step_transform(self):
        """Étape 2: Transformation des données."""
        logger.info("ÉTAPE 2/7: TRANSFORMATION DES DONNÉES")
        logger.info("-" * 70)
        
        start = time.time()
        
        # Charger données si pas déjà fait
        if not hasattr(self, 'raw_data'):
            raw_data = load_oulad_data(self.paths['raw_data'])
        else:
            raw_data = self.raw_data
        
        # Préparer dataset complet
        sample_size = self.config[self.mode]['sample_size']
        final_df = prepare_dataset(raw_data, score_deadline=self.cutoff_days, click_deadline=self.cutoff_days)
        self.final_df = final_df  # Sauvegarder pour features
        
        # Sauvegarder
        final_df.to_csv(f"{self.paths['processed_data']}/final_dataset_day{self.cutoff_days}.csv", index=False)
        
        elapsed = time.time() - start
        self.results['transform'] = {
            'duration': elapsed,
            'n_samples': len(final_df),
            'status': 'success'
        }
        
        logger.info(f"Transformation terminée en {elapsed:.2f}s")
        logger.info(f"   • Échantillons: {len(final_df)}")
        logger.info("")
    
    def _step_load(self):
        """Étape 7: Chargement dans PostgreSQL."""
        logger.info("ÉTAPE 7/7: CHARGEMENT BASE DE DONNÉES")
        logger.info("-" * 70)
        
        start = time.time()
        
        # Charger final_df si pas déjà fait
        if not hasattr(self, 'final_df'):
            import pandas as pd
            final_df = pd.read_csv(f"{self.paths['processed_data']}/final_dataset_day{self.cutoff_days}.csv")
        else:
            final_df = self.final_df
        
        # Créer connexion
        conn_string = create_connection_string()
        
        # Sauvegarder dans PostgreSQL
        save_to_postgres(final_df, 'final_dataset', conn_string)
        
        #import faetures et labels
        X = np.load(self.paths['features'])
        y = np.load(self.paths['labels'])
        
        # Charger les noms de colonnes depuis features_metadata.json
        with open(self.paths['metadata'], 'r') as f:
            metadata = json.load(f)
        column_names = metadata['column_names']
        
        logger.info(f"Chargement des features: X shape={X.shape}, y shape={y.shape}")
        logger.info(f"Nombre de colonnes: {len(column_names)}")
    
        save_features_to_postgres(X=X, y=y, column_names=column_names, connection_string=conn_string)
        with open(self.paths['model_metadata'], 'r') as f:
            metadata = json.load(f)
        save_model_metadata_to_postgres(metadata=metadata, connection_string=conn_string)
    
        
        elapsed = time.time() - start
        self.results['load'] = {
            'duration': elapsed,
            'status': 'success'
        }
        
        logger.info(f"Chargement DB terminé en {elapsed:.2f}s")
        logger.info("")
    
    def _step_features(self):
        """Étape 3: Construction des features."""
        logger.info("ÉTAPE 3/7: CONSTRUCTION DES FEATURES")
        logger.info("-" * 70)
        
        start = time.time()
        
        # Charger final_df si pas déjà fait
        if not hasattr(self, 'final_df'):
            import pandas as pd
            final_df = pd.read_csv(f"{self.paths['processed_data']}/final_dataset.csv")
        else:
            final_df = self.final_df
        
        # Construire features
        X, y, column_names, encode_dict = engineer_features(final_df)
        
        
        # Sauvegarder
        save_features(X, y, column_names, encode_dict,
                     output_dir=self.paths['processed_data'],
                     prefix=self.mode)
        
        # Renommer pour correspondre aux paths configurés
        import shutil
        shutil.move(
            f"{self.paths['processed_data']}/X_features_{self.mode}.npy",
            self.paths['features']
        )
        shutil.move(
            f"{self.paths['processed_data']}/y_labels_{self.mode}.npy",
            self.paths['labels']
        )
        shutil.move(
            f"{self.paths['processed_data']}/features_metadata_{self.mode}.json",
            self.paths['metadata']
        )
        
        elapsed = time.time() - start
        self.results['features'] = {
            'duration': elapsed,
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'status': 'success'
        }
        
        logger.info(f"Features créées en {elapsed:.2f}s")
        logger.info(f"   • Échantillons: {X.shape[0]}")
        logger.info(f"   • Features: {X.shape[1]}")
        logger.info("")
    
    def _step_train(self):
        """Étape 4: Entraînement du modèle."""
        logger.info("ÉTAPE 4/7: ENTRAÎNEMENT DU MODÈLE")
        logger.info("-" * 70)
        
        start = time.time()
        
        import numpy as np
        
        # Charger features
        X_train = np.load(self.paths['features'])
        y_train = np.load(self.paths['labels'])
        
        # Entraîner avec CV
        config = self.config[self.mode]['model_params']
        n_splits = self.config[self.mode]['n_splits_cv']
        
        model, cv_scores, model_metadata = train_with_cross_validation(
            X_train, y_train,
            self.paths['metadata'],
            config=config,
            n_splits=n_splits
        )
        
        # Sauvegarder
        save_model(model, self.paths['model'])
        save_model_metadata(model_metadata, self.paths['model_metadata'])
        
        elapsed = time.time() - start
        self.results['train'] = {
            'duration': elapsed,
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'status': 'success'
        }
        
        logger.info(f"Entraînement terminé en {elapsed:.2f}s")
        logger.info(f"   • CV Score: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")
        logger.info("")
        
        self.model = model
        self.model_metadata = model_metadata
    
    def _step_evaluate(self):
        """Étape 5: Évaluation du modèle."""
        logger.info("ÉTAPE 5/7: ÉVALUATION DU MODÈLE")
        logger.info("-" * 70)
        
        start = time.time()
        
        import numpy as np
        from src.models.train import load_model
        
        # Charger modèle et données
        model = load_model(self.paths['model'])
        X_test = np.load(self.paths['features'])[:1000]  # Échantillon
        y_test = np.load(self.paths['labels'])[:1000]
        
        # Charger feature names
        with open(self.paths['metadata'], 'r') as f:
            metadata = json.load(f)
            feature_names = metadata['column_names']
        
        # Évaluation complète
        Path(self.paths['evaluation']).mkdir(parents=True, exist_ok=True)
        
        results = evaluate_model_complete(
            model, X_test, y_test,
            feature_names=feature_names,
            save_dir=self.paths['evaluation']
        )
        
        elapsed = time.time() - start
        self.results['evaluate'] = {
            'duration': elapsed,
            'accuracy': results['accuracy'],
            'auc': results.get('auc', 0),
            'status': 'success'
        }
        
        logger.info(f"Évaluation terminée en {elapsed:.2f}s")
        logger.info(f"   • Accuracy: {results['accuracy']:.2%}")
        logger.info(f"   • AUC: {results.get('auc', 0):.4f}")
        logger.info("")
    
    def _step_predict(self):
        """Étape 6: Prédictions finales."""
        logger.info("ÉTAPE 6/7: GÉNÉRATION DES PRÉDICTIONS")
        logger.info("-" * 70)
        
        start = time.time()
        
        from src.models.train import load_model
        
        # Charger modèle
        model = load_model(self.paths['model'])
        
        # Prédictions batch
        results_df = predict_batch(
            model,
            self.paths['features'],
            output_path=self.paths['predictions']
        )
        
        elapsed = time.time() - start
        self.results['predict'] = {
            'duration': elapsed,
            'n_predictions': len(results_df),
            'status': 'success'
        }
        
        logger.info(f"Prédictions générées en {elapsed:.2f}s")
        logger.info(f"   • Nombre: {len(results_df)}")
        logger.info("")
    
    def _print_summary(self):
        """Affiche le résumé final."""
        total_time = time.time() - self.start_time
        
        logger.info("="*70)
        logger.info("PIPELINE TERMINÉ AVEC SUCCÈS!")
        logger.info("="*70)
        logger.info(f"Temps total: {total_time:.2f}s ({total_time/60:.1f} min)")
        logger.info("")
        logger.info("RÉSUMÉ PAR ÉTAPE:")
        
        for step, result in self.results.items():
            logger.info(f"    {step.upper()}: {result['duration']:.2f}s")
        
        logger.info("")
        logger.info("FICHIERS GÉNÉRÉS:")
        logger.info(f"   • Modèle: {self.paths['model']}")
        logger.info(f"   • Prédictions: {self.paths['predictions']}")
        logger.info(f"   • Évaluation: {self.paths['evaluation']}/")
        logger.info("="*70)


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description='Pipeline Orchestrator - Learning Analytics ML Pipeline'
    )
    parser.add_argument(
        '--mode',
        choices=['dev', 'prod'],
        default='dev',
        help='Mode d\'exécution (dev=développement, prod=production)'
    )
    parser.add_argument(
        '--steps',
        nargs='+',
        choices=['extract', 'transform', 'load', 'features', 'train', 'evaluate', 'predict'],
        help='Étapes spécifiques à exécuter (défaut: toutes)'
    )
    parser.add_argument(
        '--cutoff-days',
        type=int,
        default=180,
        help='Nombre de jours pour le cutoff_date (défaut: 180)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Créer et exécuter orchestrateur
    orchestrator = PipelineOrchestrator(mode=args.mode, cutoff_days=args.cutoff_days)
    orchestrator.run(steps=args.steps)


if __name__ == "__main__":
    main()
