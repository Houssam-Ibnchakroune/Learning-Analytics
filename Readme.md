# Learning Analytics - Prédiction du Risque d'Échec Étudiant

Projet de Machine Learning pour la prédiction précoce du risque d'échec des étudiants dans un environnement d'apprentissage en ligne. Ce système analyse les données d'interaction des étudiants avec la plateforme (clics, soumissions, scores) pour identifier les étudiants à risque et permettre une intervention pédagogique ciblée.

---

## Architecture du Projet

Ce projet adopte une structure professionnelle moderne pour garantir la maintenabilité, la scalabilité et la reproductibilité du code.

### Structure des Dossiers

```
prj_TD/
├── src/                          # Code source principal (installable comme package)
│   ├── data/                     # Modules ETL (Extract, Transform, Load)
│   │   ├── extract.py           # Extraction des données CSV
│   │   ├── transform.py         # Nettoyage et transformation
│   │   └── load.py              # Chargement PostgreSQL
│   ├── features/                # Feature engineering
│   │   └── build_features.py    # Création des 26 features dérivées
│   ├── models/                  # Machine Learning
│   │   ├── train.py             # Entraînement avec cross-validation
│   │   ├── evaluate.py          # Métriques et visualisations
│   │   └── predict.py           # Prédictions et scores de risque
│   └── pipeline/                # Orchestration
│       └── orchestrator.py      # Pipeline automatisé dev/prod
├── config/                      # Fichiers de configuration
├── utils/                       # Utilitaires (logging, helpers)
├── tests/                       # Tests unitaires
├── notebooks/                   # Notebooks Jupyter (version développement)
├── data/                        # Données (raw, processed, features)
├── models/                      # Modèles sauvegardés (.pkl)
├── reports/                     # Rapports et visualisations
├── pyproject.toml               # Configuration moderne du projet
├── requirements.txt             # Dépendances Python
└── .env                         # Variables d'environnement (credentials DB)
```

---

## Configuration Moderne avec pyproject.toml

Ce projet utilise le standard moderne PEP 621 pour la configuration Python via le fichier `pyproject.toml`. Cette approche offre plusieurs avantages par rapport à l'ancien `setup.py`.

### Avantages de pyproject.toml

Le fichier `pyproject.toml` centralise toute la configuration du projet dans un format TOML lisible et structuré. Il définit les métadonnées du projet, les dépendances, les outils de développement et la configuration du système de build.

**Configuration du package:**
```toml
[project]
name = "learning-analytics"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["pandas>=2.2.0", "scikit-learn>=1.4.0", ...]
```

**Configuration de la découverte des packages:**
```toml
[tool.setuptools.packages.find]
where = ["."]
include = ["src*", "config*", "utils*"]
```

Cette configuration indique à setuptools quels packages installer et où les chercher. Le paramètre `where` définit le point de départ de la recherche (ici la racine du projet), tandis que `include` liste explicitement les packages à installer (src, config, utils).

---

## Installation en Mode Éditable (pip install -e .)

L'installation en mode éditable est une technique professionnelle qui permet de développer et tester le code sans réinstallation constante.

### Fonctionnement du Mode Éditable (PEP 660)

Contrairement à une installation standard qui copie les fichiers dans `site-packages`, l'installation éditable crée un système de redirection dynamique via un Import Finder personnalisé.

**Commande d'installation:**
```bash
pip install -e .
```

**Ce qui se passe en coulisses:**

Lorsque vous exécutez cette commande, setuptools génère deux fichiers dans votre environnement virtuel:

1. Un fichier `.pth` qui active le finder au démarrage de Python
2. Un module Python `_finder.py` contenant un mapping des packages vers leurs emplacements source

**Exemple de mapping généré:**
```python
MAPPING = {
    'src': 'C:\\Users\\hibnc\\Downloads\\prj_TD\\src',
    'config': 'C:\\Users\\hibnc\\Downloads\\prj_TD\\config',
    'utils': 'C:\\Users\\hibnc\\Downloads\\prj_TD\\utils'
}
```

Quand Python rencontre `import src.data.extract`, le finder intercepte l'import et redirige vers le dossier source original plutôt que vers `site-packages`. Cela signifie que toute modification du code source est immédiatement visible sans réinstallation.

### Avantages du Mode Éditable

**Développement efficace:** Les modifications du code sont prises en compte instantanément. Vous pouvez éditer une fonction dans `src/models/train.py` et l'importer immédiatement dans un script de test sans aucune étape intermédiaire.

**Imports absolus cohérents:** Une fois installé en mode éditable, vous pouvez importer vos modules avec des chemins absolus depuis n'importe quel répertoire du système. Par exemple, `from src.models.train import train_with_cross_validation` fonctionne que vous soyez dans le dossier racine, dans `tests/` ou même dans un répertoire complètement différent.

**Reproductibilité:** Chaque développeur qui clone le projet et exécute `pip install -e .` obtient exactement la même configuration d'imports, éliminant les problèmes de `sys.path` manuel ou de `PYTHONPATH`.

---

## Pipeline ETL et Orchestration

Le projet implémente un pipeline ETL complet avec une orchestration automatisée supportant deux modes d'exécution.

### Architecture ETL

**Extract:** Chargement des fichiers CSV du dataset Open University Learning Analytics depuis le dossier `data/raw/`. Les données brutes incluent les informations étudiants, les cours, les évaluations, les interactions VLE (Virtual Learning Environment) et les résultats.

**Transform:** Nettoyage, fusion et transformation des données. Cette étape inclut la création de features agrégées comme le score moyen par date limite, le nombre total de clics, l'activité par type de ressource, et la détection des patterns d'engagement. Le pipeline génère 26 features dérivées à partir des données brutes.

**Load:** Sauvegarde des données transformées dans PostgreSQL et des features dans des fichiers `.npy` pour un accès rapide. Les métadonnées (noms de colonnes, dictionnaires d'encodage) sont sauvegardées en JSON pour garantir la reproductibilité.

### Orchestrateur Pipeline

L'orchestrateur `src/pipeline/orchestrator.py` automatise l'exécution de l'ensemble du workflow avec une interface en ligne de commande.

**Mode Développement (dev):**
```bash
python -m src.pipeline.orchestrator --mode dev --cutoff-days 180
```

En mode développement, le pipeline travaille sur un échantillon réduit de 5000 lignes pour des itérations rapides. La cross-validation utilise 3 folds et le modèle Random Forest est configuré avec 100 arbres. Les fichiers générés sont suffixés `_dev` pour éviter toute confusion avec les versions production.

**Mode Production (prod):**
```bash
python -m src.pipeline.orchestrator --mode prod --cutoff-days 180
```

Le mode production traite l'intégralité du dataset, utilise une cross-validation à 5 folds pour une évaluation robuste, et configure Random Forest avec 200 arbres pour maximiser la performance. Les fichiers sont suffixés `_prod`.

**Exécution d'étapes spécifiques:**
```bash
python -m src.pipeline.orchestrator --mode dev --steps features train evaluate
```

Cette flexibilité permet de ré-exécuter seulement les étapes nécessaires après une modification de code ou de configuration, économisant du temps lors du développement itératif.

**Paramètre cutoff-days:**

Le paramètre `--cutoff-days` contrôle le nombre de jours après le début du cours utilisé pour calculer les features. Par exemple, avec `--cutoff-days 90`, seules les interactions des 90 premiers jours sont utilisées pour la prédiction, permettant d'intervenir plus tôt. Le nom du modèle sauvegardé reflète cette configuration: `random_forest_day90_dev.pkl`.

---

## Machine Learning

Le système utilise Random Forest pour la classification binaire (réussite/échec) avec une pipeline complète d'entraînement, évaluation et prédiction.

### Entraînement avec Cross-Validation

L'entraînement utilise K-Fold cross-validation pour évaluer la généralisation du modèle de manière robuste. La validation croisée divise les données en K folds (3 en dev, 5 en prod), entraîne sur K-1 folds et valide sur le fold restant, répétant l'opération K fois.

**Résultats obtenus:**
- Accuracy moyenne: 93.6%
- Écart-type: 0.37%
- AUC: 0.998

Le modèle détecte 100% des étudiants en échec (recall classe Fail = 1.0), ce qui est crucial pour un système d'intervention précoce où manquer un étudiant à risque a un coût élevé.

### Features Engineering

Le système génère 26 features dérivées à partir des données brutes, incluant:

**Features temporelles:** Score moyen au jour 90, activité VLE avant différentes deadlines, délai moyen de soumission

**Features d'engagement:** Nombre total de clics, clics par type de ressource (homepage, forum, quiz, resource, url), activité par module

**Features académiques:** Crédits étudiés précédemment, nombre d'inscriptions antérieures, région géographique

**Features démographiques:** Catégories d'âge, niveau d'éducation, indice de déprivation (IMD)

L'importance des features révèle que `mean_score_day90` (26% d'importance) est le prédicteur le plus fort, suivi par `homepage` (8.5%), `forumng` (8.2%) et `quiz` (7.7%). Cette cohérence entre l'importance des features et l'intuition pédagogique valide la pertinence du modèle.

### Évaluation et Visualisations

Le module `src/models/evaluate.py` génère des rapports complets avec matrice de confusion, courbe ROC, et importance des features. Les visualisations sont sauvegardées dans `reports/figures/` pour une analyse détaillée et une présentation des résultats.

### Prédictions avec Scores de Risque

Le système produit non seulement une classification binaire mais aussi un score de risque continu entre 0 et 1, permettant une priorisation des interventions. Les étudiants sont catégorisés en trois niveaux de risque: Faible (< 30%), Moyen (30-70%), Élevé (> 70%).

---

## Base de Données PostgreSQL

Le projet utilise PostgreSQL pour la persistance des données transformées, permettant des requêtes SQL complexes et une intégration facilitée avec des outils de Business Intelligence comme Power BI.

### Configuration Sécurisée

Les credentials de la base de données sont stockés dans un fichier `.env` qui n'est jamais commité dans Git grâce au `.gitignore`. Un fichier `.env.example` fournit le template:

```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=learning_analytics
DB_USER=postgres
DB_PASSWORD=your_password
```

Le module `src/data/load.py` charge ces variables via `python-dotenv` et crée automatiquement la connection string SQLAlchemy.

### Tables Créées

Le pipeline crée plusieurs tables pour stocker les données à différents niveaux de granularité, facilitant l'analyse exploratoire et la création de dashboards. La table principale `final_dataset` contient les features calculées prêtes pour le machine learning.

---

## Gestion des Environnements Python

Le projet utilise un environnement virtuel Python isolé pour garantir la reproductibilité des dépendances.

### Installation de l'Environnement

**Création de l'environnement virtuel:**
```bash
python -m venv venv
```

**Activation (Windows PowerShell):**
```bash
.\venv\Scripts\activate
```

**Installation des dépendances:**
```bash
pip install -r requirements.txt
```

**Installation du projet en mode éditable:**
```bash
pip install -e .
```

Cette dernière commande est cruciale: elle configure le système d'imports pour permettre `from src.models.train import ...` depuis n'importe où dans le projet, éliminant les manipulations de `sys.path` ou `PYTHONPATH`.

---

## Notebooks Jupyter (Version Développement)

Le dossier `notebooks/` contient quatre notebooks Jupyter qui représentent la phase initiale de développement et d'exploration:

**01_data_exploration.ipynb:** Analyse exploratoire du dataset, statistiques descriptives, visualisations de distributions, détection de valeurs manquantes et outliers

**02_feature_engineering.ipynb:** Expérimentation avec différentes features, tests de corrélation, analyse de l'importance des variables

**03_model_training.ipynb:** Tests de différents algorithmes, tuning d'hyperparamètres, comparaison de performances

**04_evaluation.ipynb:** Visualisations détaillées des résultats, analyse des erreurs, interprétation du modèle

Ces notebooks sont des versions de développement et d'expérimentation. Une fois les approches validées dans les notebooks, le code est réorganisé en modules Python réutilisables dans `src/` pour créer un pipeline automatisé et production-ready. Les notebooks restent utiles pour la documentation et la démonstration des décisions méthodologiques prises pendant le projet.

---

## Tests Automatisés

Le projet inclut des tests unitaires pour valider le bon fonctionnement des modules principaux.

**Tests disponibles:**

`tests/test_train_process.py`: Validation de l'entraînement du modèle, vérification de la cross-validation, sauvegarde des modèles et métadonnées

`tests/test_evaluate_process.py`: Validation des métriques de classification, génération des visualisations, calcul de l'importance des features

`tests/test_predict_process.py`: Validation des prédictions, calcul des scores de risque, catégorisation des niveaux de risque

**Exécution des tests:**
```bash
python -m tests.test_train_process
python -m tests.test_evaluate_process
python -m tests.test_predict_process
```

Chaque test produit des logs détaillés confirmant que toutes les fonctions s'exécutent correctement et génèrent les résultats attendus.

---

## Logging Professionnel

Le projet implémente un système de logging structuré configuré dans `utils/login_setup.py`. Les logs sont affichés en console avec des couleurs pour faciliter la lecture (INFO en vert, WARNING en jaune, ERROR en rouge) et sauvegardés dans des fichiers dans le dossier `logs/`.

Chaque module utilise son propre logger:
```python
import logging
logger = logging.getLogger(__name__)
```

Cette approche permet de tracer précisément l'origine de chaque message et de filtrer les logs par module si nécessaire.

---

## Utilisation du Projet

### Exécution Complète du Pipeline

**Développement rapide (2 minutes):**
```bash
python -m src.pipeline.orchestrator --mode dev --cutoff-days 180
```

**Production complète (7 minutes):**
```bash
python -m src.pipeline.orchestrator --mode prod --cutoff-days 180
```

### Exécution d'Étapes Spécifiques

**Re-entraîner le modèle avec nouveaux hyperparamètres:**
```bash
python -m src.pipeline.orchestrator --mode dev --steps train evaluate
```

**Générer de nouvelles prédictions:**
```bash
python -m src.pipeline.orchestrator --mode prod --steps predict
```

**Tester différents cutoff_days:**
```bash
python -m src.pipeline.orchestrator --mode dev --cutoff-days 90
python -m src.pipeline.orchestrator --mode dev --cutoff-days 120
```

### Fichiers Générés

Après exécution, le pipeline génère:

**Modèles:** `models/random_forest_day180_dev.pkl` et métadonnées JSON correspondantes

**Prédictions:** `reports/predictions_dev.csv` avec colonnes prediction, risk_score, risk_level

**Visualisations:** Matrice de confusion, courbe ROC, importance des features dans `reports/figures_dev/`

**Features:** Arrays NumPy des features et labels dans `data/processed/`

---

## Résultats et Performance

Le modèle atteint des performances élevées sur la tâche de prédiction du risque d'échec:

**Métriques globales:**
- Accuracy: 93.6%
- Precision: 94.0%
- Recall: 93.6%
- F1-Score: 93.4%
- AUC-ROC: 0.998

**Performance par classe:**

Classe Pass (Réussite):
- Precision: 99% (très peu de faux positifs)
- Recall: 78% (22% de réussites prédites comme échec)

Classe Fail (Échec):
- Precision: 92% (8% de faux positifs)
- Recall: 100% (aucun échec manqué)

Le modèle est optimisé pour détecter tous les étudiants en échec au prix d'un surdiagnostic acceptable pour les réussites. Cette configuration est idéale pour un système d'intervention précoce où le coût de manquer un étudiant à risque est élevé.

---

## Technologies et Dépendances

**Python:** 3.12

**Data Processing:** pandas 2.2, numpy 1.26

**Machine Learning:** scikit-learn 1.4, xgboost 2.0, imbalanced-learn 0.12

**Database:** PostgreSQL via SQLAlchemy 2.0, psycopg2-binary 2.9

**Visualization:** matplotlib 3.8, seaborn 0.13

**Development:** jupyter 1.0, pytest 7.4, python-dotenv

**Configuration:** setuptools 68+, wheel (PEP 660 editable install)

---

