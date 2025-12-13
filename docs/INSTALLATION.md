# Guide d'Installation - OULAD Learning Analytics

## Prérequis

- Python 3.8 ou supérieur
- PostgreSQL 12 ou supérieur
- Git (pour cloner le projet)

---

## Installation de PostgreSQL

### Windows

1. **Télécharger PostgreSQL:**
   - https://www.postgresql.org/download/windows/
   - Télécharger l'installateur (version 14 ou plus récente)

2. **Installer:**
   - Exécuter l'installateur
   - Définir un mot de passe pour l'utilisateur `postgres`
   - **Noter ce mot de passe** (vous en aurez besoin !)
   - Port par défaut : 5432

3. **Vérifier l'installation:**
   ```powershell
   psql --version
   ```

### Alternative: Docker (recommandé pour le développement)

```powershell
# Télécharger et démarrer PostgreSQL avec Docker
docker run --name postgres-oulad -e POSTGRES_PASSWORD=postgres -p 5432:5432 -d postgres:14
```

---

## Installation du Projet Python

### Cloner le projet

```bash
git clone https://github.com/Houssam-Ibnchakroune/Learning-Analytics.git
cd Learning-Analytics
```

### Créer un environnement virtuel

```powershell
# Créer l'environnement
python -m venv venv

# Activer l'environnement
.\venv\Scripts\Activate.ps1
```

### Installer les dépendances

```powershell
pip install -r requirements.txt
```

---

## Configuration de la Base de Données

### Créer le fichier .env

```powershell
# Copier le template
copy .env.example .env
```

### Éditer .env avec vos identifiants

Ouvrez `.env` et modifiez :

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=oulad_db
DB_USER=postgres
DB_PASSWORD=VOTRE_MOT_DE_PASSE_ICI  # Changez ceci !
```

### Exécuter le script de configuration

```powershell
python setup_database.py
```

**Ce script va :**
- Créer la base de données `oulad_db`
- Créer toutes les tables nécessaires
- Créer les index pour les performances
- Vérifier que tout est configuré correctement

**Sortie attendue :**
```
============================================================
  CONFIGURATION DE LA BASE DE DONNÉES OULAD
============================================================
 Création de la base de données...

 Base de données 'oulad_db' créée avec succès!

 Création des tables...

 Tables créées avec succès:
   - students_prepared
   - ml_features
   - predictions
   - model_metadata
   - model_metrics
   - student_interventions

 Création des index...

 Index créés avec succès

 Vérification de la configuration...

 Tables trouvées dans 'oulad_db':
   - ml_features                      (0 lignes)
   - model_metadata                   (0 lignes)
   - model_metrics                    (0 lignes)
   - predictions                      (0 lignes)
   - student_interventions            (0 lignes)
   - students_prepared                (0 lignes)

============================================================
   CONFIGURATION TERMINÉE AVEC SUCCÈS!
============================================================
```

---

## Vérification de l'Installation

### Test de connexion PostgreSQL

```powershell
python -m src.data.load
```

**Sortie attendue :**
```
Connexion PostgreSQL réussie!
   Version: PostgreSQL 14.x ...
```

### Test des modules ETL

```powershell
# Test extraction
python -c "from src.data.extract import load_oulad_data; data = load_oulad_data('data/raw/open+university+learning+analytics+dataset/'); print('Extract OK')"

# Test transformation
python -c "from src.data.transform import prepare_dataset; print('Transform OK')"

# Test features
python -c "from src.features.build_features import engineer_features; print('Features OK')"
```

---

## Télécharger les Données OULAD

### Télécharger le dataset

1. **URL:** https://analyse.kmi.open.ac.uk/open_dataset
2. Télécharger le fichier ZIP
3. Extraire dans : `data/raw/open+university+learning+analytics+dataset/`

**Structure attendue :**
```
data/
└── raw/
    └── open+university+learning+analytics+dataset/
        ├── assessments.csv
        ├── courses.csv
        ├── studentAssessment.csv
        ├── studentInfo.csv
        ├── studentRegistration.csv
        ├── studentVle.csv
        └── vle.csv
```

---

## Exécuter le Pipeline Complet

```powershell
# À venir : pipeline.py orchestrera tout le processus
python pipeline.py
```

---

## Dépannage

### Erreur: "psycopg2 installation failed"

**Solution Windows:**
```powershell
pip install psycopg2-binary
```

### Erreur: "connexion refused" PostgreSQL

**Vérifications:**
1. PostgreSQL est démarré :
   ```powershell
   # Vérifier le service
   Get-Service postgresql*
   ```

2. Port 5432 est disponible :
   ```powershell
   netstat -an | findstr 5432
   ```

3. Identifiants corrects dans `.env`

### Erreur: "database already exists"

C'est normal si vous réexécutez le script. Il détecte que la DB existe et continue.

### Erreur: "permission denied"

Vérifiez que l'utilisateur PostgreSQL a les droits :
```sql
-- Se connecter à PostgreSQL
psql -U postgres

-- Donner tous les droits
GRANT ALL PRIVILEGES ON DATABASE oulad_db TO postgres;
```

---

## Structure du Projet

```
prj_TD/
├── .env                        # Configuration (non commité)
├── .env.example                # Template de configuration
├── requirements.txt            # Dépendances Python
├── setup_database.py           # Script de création DB
├── pipeline.py                 # Pipeline complet (à venir)
├── data/
│   ├── raw/                    # Données brutes OULAD
│   └── processed/              # Données préparées
├── src/
│   ├── data/
│   │   ├── extract.py         # Extraction CSV
│   │   ├── transform.py       # Transformation
│   │   └── load.py            # Chargement PostgreSQL
│   ├── features/
│   │   └── build_features.py  # Feature engineering
│   └── models/
│       ├── train.py           # Entraînement
│       ├── predict.py         # Prédictions
│       └── evaluate.py        # Évaluation
├── notebooks/                  # Jupyter notebooks
└── tests/                      # Tests unitaires
```

---

## Checklist d'Installation

- [ ] PostgreSQL installé et démarré
- [ ] Python 3.8+ installé
- [ ] Environnement virtuel créé et activé
- [ ] Dépendances installées (`pip install -r requirements.txt`)
- [ ] Fichier `.env` créé avec les bons identifiants
- [ ] Script `setup_database.py` exécuté avec succès
- [ ] Données OULAD téléchargées et extraites
- [ ] Tests de connexion réussis


