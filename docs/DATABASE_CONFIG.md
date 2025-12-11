# Configuration de la Base de Données PostgreSQL

## Sécurité et Variables d'Environnement

Ce projet utilise des **variables d'environnement** pour stocker les identifiants de la base de données de manière sécurisée.

---

## Installation

### 1. Installer python-dotenv

```bash
pip install python-dotenv
```

### 2. Créer le fichier .env

Copiez le fichier d'exemple et remplissez avec vos vraies valeurs :

```bash
cp .env.example .env
```

### 3. Éditer .env

Ouvrez `.env` et remplissez avec vos identifiants PostgreSQL :

```env
# Configuration PostgreSQL
DB_HOST=localhost
DB_PORT=5432
DB_NAME=oulad_db
DB_USER=postgres
DB_PASSWORD=votre_mot_de_passe_ici
```

**IMPORTANT:** Le fichier `.env` est dans `.gitignore` et ne sera **jamais** commité sur Git !

---

##  Utilisation

### Option 1 : Utiliser .env (Recommandé)

```python
from src.data.load import create_connection_string, save_to_postgres

# La connexion utilise automatiquement .env
conn_str = create_connection_string()

# Sauvegarder un DataFrame
save_to_postgres(df, 'students', conn_str)
```

### Option 2 : Variables d'environnement système

Définissez les variables avant d'exécuter le script :

**Windows (PowerShell):**
```powershell
$env:DB_HOST="localhost"
$env:DB_PASSWORD="votre_mot_de_passe"
python pipeline.py
```

**Linux/Mac:**
```bash
export DB_HOST="localhost"
export DB_PASSWORD="votre_mot_de_passe"
python pipeline.py
```

### Option 3 : Spécifier manuellement (Non recommandé pour production)

```python
conn_str = create_connection_string(
    host='localhost',
    database='oulad_db',
    user='postgres',
    password='secret',
    use_env=False  # Désactive .env
)
```

---

##  Configuration PostgreSQL

### Créer la base de données

```sql
-- Connexion à PostgreSQL
psql -U postgres

-- Créer la base de données
CREATE DATABASE oulad_db;

-- Vérifier
\l
```

### Tester la connexion

```python
from src.data.load import test_connection, create_connection_string

conn_str = create_connection_string()
test_connection(conn_str)
```


---

##  Dépannage

### Erreur: "DB_PASSWORD non défini"

**Solution:** Créez le fichier `.env` avec vos identifiants

### Erreur: "python-dotenv non installé"

**Solution:** 
```bash
pip install python-dotenv
```

### Erreur de connexion PostgreSQL

**Vérifications:**
1. PostgreSQL est installé et démarré
2. Les identifiants dans `.env` sont corrects
3. La base de données `oulad_db` existe
4. Le port 5432 est disponible

---

##  Structure des Fichiers

```
prj_TD/
├── .env                  # Vos identifiants (NON commité)
├── .env.example          # Template (commité sur Git)
├── .gitignore            # Contient .env
└── src/
    └── data/
        └── load.py       # Utilise les variables d'environnement
```

---


- [python-dotenv Documentation](https://github.com/theskumar/python-dotenv)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
