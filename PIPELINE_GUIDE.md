# 🔄 Guide du Pipeline Orchestrator

## 📁 Emplacement
```
src/pipeline/orchestrator.py
```

---

## 🚀 Exécution

### **MODE DÉVELOPPEMENT** (Recommandé pour tests)
```bash
# Pipeline complet
python -m src.pipeline.orchestrator --mode dev

# Étapes spécifiques
python -m src.pipeline.orchestrator --mode dev --steps features train evaluate
```

**Caractéristiques Dev:**
- ✅ Échantillon réduit (5000 lignes)
- ✅ CV rapide (3 folds)
- ✅ Modèle léger (100 estimators)
- ✅ Fichiers suffixés `_dev`

---

### **MODE PRODUCTION** (Données complètes)
```bash
# Pipeline complet
python -m src.pipeline.orchestrator --mode prod

# Étapes spécifiques
python -m src.pipeline.orchestrator --mode prod --steps train evaluate predict
```

**Caractéristiques Prod:**
- ✅ Toutes les données
- ✅ CV robuste (5 folds)
- ✅ Modèle optimisé (200 estimators)
- ✅ Fichiers suffixés `_prod`

---

## 📋 Étapes Disponibles

| Étape | Description | Temps (dev) | Temps (prod) |
|-------|-------------|-------------|--------------|
| `extract` | Extraction des CSV | ~5s | ~10s |
| `transform` | Nettoyage des données | ~10s | ~30s |
| `load` | Chargement PostgreSQL | ~15s | ~60s |
| `features` | Feature engineering | ~20s | ~120s |
| `train` | Entraînement + CV | ~30s | ~180s |
| `evaluate` | Métriques + graphiques | ~10s | ~20s |
| `predict` | Prédictions batch | ~5s | ~15s |

**Total:** ~2 min (dev) | ~7 min (prod)

---

## 🎯 Cas d'Usage

### **1. Premier lancement complet**
```bash
python -m src.pipeline.orchestrator --mode dev
```
→ Exécute toutes les étapes (extract → predict)

---

### **2. Re-entraîner le modèle uniquement**
```bash
python -m src.pipeline.orchestrator --mode dev --steps train evaluate
```
→ Utile après modification des hyperparamètres

---

### **3. Nouvelles prédictions seulement**
```bash
python -m src.pipeline.orchestrator --mode prod --steps predict
```
→ Utilise le modèle existant pour prédire

---

### **4. Pipeline complet production**
```bash
python -m src.pipeline.orchestrator --mode prod
```
→ Pipeline complet avec toutes les données

---

## 📁 Fichiers Générés

### **Mode Dev:**
```
data/processed/
├── features_dev.npy
├── labels_dev.npy
└── features_metadata_dev.json

models/
├── random_forest_dev.pkl
└── model_metadata_dev.json

reports/
├── predictions_dev.csv
└── figures_dev/
    ├── confusion_matrix.png
    ├── roc_curve.png
    └── feature_importance.png
```

### **Mode Prod:**
```
(même structure avec suffix _prod)
```

---

## 🔧 Configuration Personnalisée

Modifier dans `orchestrator.py`:

```python
def _get_default_config(self):
    return {
        'dev': {
            'sample_size': 5000,  # Nombre d'échantillons
            'n_splits_cv': 3,     # Folds CV
            'model_params': {
                'n_estimators': 100,  # Arbres RF
                ...
            }
        },
        'prod': {
            'sample_size': None,  # Toutes les données
            'n_splits_cv': 5,
            'model_params': {
                'n_estimators': 200,
                ...
            }
        }
    }
```

---

## 📊 Exemple de Sortie

```
======================================================================
🚀 PIPELINE ORCHESTRATOR - MODE: DEV
======================================================================
📋 Étapes à exécuter: extract, transform, load, features, train, evaluate, predict

🔵 ÉTAPE 1/7: EXTRACTION DES DONNÉES
----------------------------------------------------------------------
✅ Extraction terminée en 4.23s

🔵 ÉTAPE 2/7: TRANSFORMATION DES DONNÉES
----------------------------------------------------------------------
✅ Transformation terminée en 8.15s

...

======================================================================
🎉 PIPELINE TERMINÉ AVEC SUCCÈS!
======================================================================
⏱️  Temps total: 127.45s (2.1 min)

📊 RÉSUMÉ PAR ÉTAPE:
   ✅ EXTRACT: 4.23s
   ✅ TRANSFORM: 8.15s
   ✅ LOAD: 12.34s
   ✅ FEATURES: 23.45s
   ✅ TRAIN: 34.56s
   ✅ EVALUATE: 15.67s
   ✅ PREDICT: 6.78s

📁 FICHIERS GÉNÉRÉS:
   • Modèle: models/random_forest_dev.pkl
   • Prédictions: reports/predictions_dev.csv
   • Évaluation: reports/figures_dev/
======================================================================
```

---

## 🐛 Dépannage

### **Erreur: Module not found**
```bash
# Vérifier l'installation editable
pip install -e .
```

### **Erreur: Database connection**
```bash
# Vérifier .env
cat .env

# Tester connexion
python -c "from src.data.load import create_connection_string; print(create_connection_string())"
```

### **Erreur: File not found**
```bash
# Vérifier structure
ls data/raw/
ls data/processed/
```

---

## ✅ Checklist avant Production

- [ ] Tester en mode `dev` d'abord
- [ ] Vérifier `.env` avec credentials production
- [ ] Backup de la base de données
- [ ] Espace disque suffisant (~500MB)
- [ ] Vérifier logs dans `logs/pipeline_prod.log`

---

## 🎓 Bonnes Pratiques

1. **Toujours tester en dev avant prod**
2. **Versionner les modèles** (ajouter timestamp au nom)
3. **Monitorer les logs** pendant l'exécution
4. **Sauvegarder les résultats prod** régulièrement
5. **Documenter les changements** de config

---

**Bon pipeline! 🚀**
