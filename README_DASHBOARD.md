# 📊 Learning Analytics Dashboard - Streamlit

Dashboard interactif pour visualiser les prédictions de risque d'échec étudiant.

## 🚀 Installation

1. **Installer les dépendances:**

```bash
pip install -r requirements_dashboard.txt
```

Ou installer individuellement:

```bash
pip install streamlit pandas plotly numpy matplotlib seaborn
```

## ▶️ Lancement du Dashboard

Depuis la racine du projet, exécuter:

```bash
streamlit run src/visualization/streamlit_dashboard.py
```

Le dashboard s'ouvrira automatiquement dans votre navigateur par défaut à l'adresse:
```
http://localhost:8501
```

## 📁 Fichiers Nécessaires

Le dashboard utilise les fichiers CSV suivants (générés par le pipeline):

- `reports/predictions_dev.csv` - Prédictions et scores de risque
- `data/processed/final_dataset_day180.csv` - Dataset complet
- `reports/figures/feature_importance.csv` - Importance des features

## 🎯 Fonctionnalités

### 📈 **Page 1: Vue Executive**
- KPIs principaux (Total étudiants, Risque élevé, Score moyen)
- Distribution des niveaux de risque (graphique en anneau)
- Top 10 modules par risque
- Distribution des scores de risque

### 👥 **Page 2: Analyse Démographique**
- Risque par tranche d'âge
- Risque par genre
- Risque par région
- Matrice Région × Niveau d'éducation

### 🎯 **Page 3: Importance des Features**
- Top 15 features les plus importantes
- Graphique d'importance
- Contribution cumulative

### ⚠️ **Page 4: Liste d'Intervention**
- Liste des étudiants à risque
- Filtrage par urgence (>85%)
- Export CSV
- Scatter plot engagement vs performance

## 🔍 Filtres Interactifs

Dans la sidebar, vous pouvez filtrer par:
- Module
- Niveau de risque
- Genre
- Région

## 📥 Export de Données

Sur la page "Liste d'Intervention", vous pouvez télécharger la liste des étudiants à risque au format CSV.

## 🎨 Palette de Couleurs

- 🟢 Risque Faible: Vert (#388E3C)
- 🟠 Risque Moyen: Orange (#FF9800)
- 🔴 Risque Élevé: Rouge (#D32F2F)

## 💡 Conseils d'Utilisation

1. **Commencez par la Vue Executive** pour avoir une vision globale
2. **Utilisez les filtres** dans la sidebar pour analyser des segments spécifiques
3. **Page Démographique** pour identifier les groupes à risque
4. **Page Intervention** pour actions concrètes sur les étudiants

## 🐛 Dépannage

Si le dashboard ne se lance pas:

1. Vérifiez que tous les fichiers CSV existent
2. Vérifiez les chemins relatifs (le script doit être lancé depuis la racine)
3. Assurez-vous que toutes les dépendances sont installées

## 📊 Comparaison avec Power BI

Ce dashboard Streamlit reproduit toutes les visualisations décrites pour Power BI:
- ✅ KPIs et métriques
- ✅ Graphiques interactifs
- ✅ Filtres dynamiques
- ✅ Export de données
- ✅ Analyses démographiques
- ✅ Features importance

**Avantages de Streamlit:**
- Gratuit et open-source
- Facilement personnalisable (code Python)
- Déployable sur le cloud (Streamlit Cloud)
- Intégration directe avec le pipeline Python

## 🌐 Déploiement

Pour déployer le dashboard en ligne (gratuit):

1. Créer un compte sur [Streamlit Cloud](https://streamlit.io/cloud)
2. Connecter votre repository GitHub
3. Sélectionner `src/visualization/streamlit_dashboard.py`
4. Déployer!

---

**Développé pour le projet Learning Analytics - Prédiction du Risque d'Échec Étudiant**
