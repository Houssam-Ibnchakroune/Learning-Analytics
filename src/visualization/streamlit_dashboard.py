"""
Dashboard Streamlit - Learning Analytics
Visualisation interactive des prédictions de risque d'échec étudiant
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from pathlib import Path

# Configuration de la page
st.set_page_config(
    page_title="Learning Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Fonction de chargement des données avec cache
@st.cache_data
def load_data():
    """Charge tous les fichiers CSV nécessaires"""
    base_path = Path(__file__).parent.parent.parent
    
    # Chemins des fichiers
    predictions_path = base_path / "reports" / "predictions_prod.csv"
    dataset_path = base_path / "data" / "processed" / "final_dataset_day180.csv"
    features_path = base_path / "reports" / "figures_prod" / "feature_importance.csv"
    
    # Chargement
    predictions = pd.read_csv(predictions_path)
    dataset = pd.read_csv(dataset_path)
    features = pd.read_csv(features_path)
    
    # Ajouter un index pour jointure
    predictions['ID_Etudiant'] = range(1, len(predictions) + 1)
    dataset['ID_Etudiant'] = range(1, len(dataset) + 1)
    
    # Joindre les données
    df_merged = dataset.merge(predictions, on='ID_Etudiant', how='left')
    
    return df_merged, features

# Chargement des données
try:
    df, features_df = load_data()
    
    # Titre principal
    st.title("📊 Learning Analytics Dashboard")
    st.markdown("### Prédiction du Risque d'Échec Étudiant")
    st.markdown("---")
    
    # Sidebar - Filtres
    st.sidebar.header("🔍 Filtres")
    
    # Filtre par module
    modules = ['Tous'] + sorted(df['code_module'].unique().tolist())
    selected_module = st.sidebar.selectbox("Module", modules)
    
    # Filtre par niveau de risque
    risk_levels = ['Tous'] + sorted(df['risk_level'].unique().tolist())
    selected_risk = st.sidebar.selectbox("Niveau de Risque", risk_levels)
    
    # Filtre par genre
    genders = ['Tous'] + sorted(df['gender'].unique().tolist())
    selected_gender = st.sidebar.selectbox("Genre", genders)
    
    # Filtre par région
    regions = ['Tous'] + sorted(df['region'].unique().tolist())
    selected_region = st.sidebar.selectbox("Région", regions)
    
    # Appliquer les filtres
    df_filtered = df.copy()
    if selected_module != 'Tous':
        df_filtered = df_filtered[df_filtered['code_module'] == selected_module]
    if selected_risk != 'Tous':
        df_filtered = df_filtered[df_filtered['risk_level'] == selected_risk]
    if selected_gender != 'Tous':
        df_filtered = df_filtered[df_filtered['gender'] == selected_gender]
    if selected_region != 'Tous':
        df_filtered = df_filtered[df_filtered['region'] == selected_region]
    
    st.sidebar.markdown(f"**{len(df_filtered):,}** étudiants filtrés")
    
    # Onglets principaux
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Vue Executive",
        "👥 Analyse Démographique", 
        "🎯 Importance Features",
        "⚠️ Liste d'Intervention"
    ])
    
    # ==================== TAB 1: VUE EXECUTIVE ====================
    with tab1:
        st.header("Vue d'ensemble Executive")
        
        # KPIs en haut
        col1, col2, col3, col4 = st.columns(4)
        
        total_students = len(df_filtered)
        high_risk = len(df_filtered[df_filtered['risk_level'] == 'Élevé'])
        avg_risk_score = df_filtered['risk_score'].mean() * 100
        fail_predicted = len(df_filtered[df_filtered['prediction'] == 1])
        
        with col1:
            st.metric(
                label="👨‍🎓 Total Étudiants",
                value=f"{total_students:,}",
                delta=None
            )
        
        with col2:
            st.metric(
                label="⚠️ Risque Élevé",
                value=f"{high_risk:,}",
                delta=f"{(high_risk/total_students*100):.1f}%"
            )
        
        with col3:
            st.metric(
                label="📊 Score Risque Moyen",
                value=f"{avg_risk_score:.1f}%",
                delta=None
            )
        
        with col4:
            st.metric(
                label="❌ Échec Prédit",
                value=f"{fail_predicted:,}",
                delta=f"{(fail_predicted/total_students*100):.1f}%"
            )
        
        st.markdown("---")
        
        # Graphiques principaux
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Distribution des Niveaux de Risque")
            
            # Graphique en anneau
            risk_counts = df_filtered['risk_level'].value_counts()
            
            fig_donut = go.Figure(data=[go.Pie(
                labels=risk_counts.index,
                values=risk_counts.values,
                hole=0.4,
                marker=dict(colors=['#388E3C', '#FF9800', '#D32F2F'])
            )])
            
            fig_donut.update_layout(
                showlegend=True,
                height=350,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig_donut, use_container_width=True)
        
        with col2:
            st.subheader("Top 10 Modules par Risque Élevé")
            
            # Compter les risques élevés par module
            module_risk = df_filtered[df_filtered['risk_level'] == 'Élevé'].groupby('code_module').size().reset_index(name='count')
            module_risk = module_risk.sort_values('count', ascending=False).head(10)
            
            fig_bars = px.bar(
                module_risk,
                x='code_module',
                y='count',
                color='count',
                color_continuous_scale='Reds',
                labels={'code_module': 'Module', 'count': 'Étudiants à Risque Élevé'}
            )
            
            fig_bars.update_layout(
                height=350,
                showlegend=False,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig_bars, use_container_width=True)
        
        # Graphique évolution du score de risque
        st.subheader("Distribution des Scores de Risque")
        
        fig_hist = px.histogram(
            df_filtered,
            x='risk_score',
            nbins=50,
            color='risk_level',
            color_discrete_map={'Faible': '#388E3C', 'Moyen': '#FF9800', 'Élevé': '#D32F2F'},
            labels={'risk_score': 'Score de Risque', 'count': 'Nombre d\'Étudiants'}
        )
        
        fig_hist.update_layout(height=350)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # ==================== TAB 2: ANALYSE DÉMOGRAPHIQUE ====================
    with tab2:
        st.header("Analyse Démographique des Étudiants à Risque")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Risque par Tranche d'Âge")
            
            # Barres empilées par âge
            age_risk = pd.crosstab(df_filtered['age_band'], df_filtered['risk_level'])
            
            fig_age = go.Figure()
            colors_map = {'Faible': '#388E3C', 'Moyen': '#FF9800', 'Élevé': '#D32F2F'}
            
            for risk_level in ['Faible', 'Moyen', 'Élevé']:
                if risk_level in age_risk.columns:
                    fig_age.add_trace(go.Bar(
                        name=risk_level,
                        x=age_risk.index,
                        y=age_risk[risk_level],
                        marker_color=colors_map[risk_level]
                    ))
            
            fig_age.update_layout(
                barmode='stack',
                height=350,
                xaxis_title="Tranche d'Âge",
                yaxis_title="Nombre d'Étudiants",
                legend_title="Niveau de Risque"
            )
            
            st.plotly_chart(fig_age, use_container_width=True)
        
        with col2:
            st.subheader("Risque par Genre")
            
            gender_risk = pd.crosstab(df_filtered['gender'], df_filtered['risk_level'])
            
            fig_gender = go.Figure()
            
            for risk_level in ['Faible', 'Moyen', 'Élevé']:
                if risk_level in gender_risk.columns:
                    fig_gender.add_trace(go.Bar(
                        name=risk_level,
                        x=gender_risk.index,
                        y=gender_risk[risk_level],
                        marker_color=colors_map[risk_level]
                    ))
            
            fig_gender.update_layout(
                barmode='group',
                height=350,
                xaxis_title="Genre",
                yaxis_title="Nombre d'Étudiants",
                legend_title="Niveau de Risque"
            )
            
            st.plotly_chart(fig_gender, use_container_width=True)
        
        # Région
        st.subheader("Risque par Région")
        
        region_risk = df_filtered.groupby(['region', 'risk_level']).size().reset_index(name='count')
        
        fig_region = px.bar(
            region_risk,
            x='region',
            y='count',
            color='risk_level',
            color_discrete_map={'Faible': '#388E3C', 'Moyen': '#FF9800', 'Élevé': '#D32F2F'},
            barmode='stack',
            labels={'region': 'Région', 'count': 'Nombre d\'Étudiants'}
        )
        
        fig_region.update_layout(
            height=400,
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig_region, use_container_width=True)
        
        # Matrice Région × Niveau d'éducation
        st.subheader("Matrice Région × Niveau d'Éducation (% Risque Élevé)")
        
        # Calculer le pourcentage de risque élevé
        pivot_data = df_filtered.groupby(['region', 'highest_education']).apply(
            lambda x: (x['risk_level'] == 'Élevé').sum() / len(x) * 100
        ).reset_index(name='pct_high_risk')
        
        pivot_table = pivot_data.pivot(index='region', columns='highest_education', values='pct_high_risk')
        
        fig_heatmap = px.imshow(
            pivot_table,
            labels=dict(x="Niveau d'Éducation", y="Région", color="% Risque Élevé"),
            color_continuous_scale='Reds',
            aspect='auto'
        )
        
        fig_heatmap.update_layout(height=500)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # ==================== TAB 3: IMPORTANCE FEATURES ====================
    with tab3:
        st.header("Importance des Features du Modèle")
        
        st.info("🔍 Ces features ont le plus d'impact sur la prédiction du risque d'échec")
        
        # Top 15 features
        features_top15 = features_df.head(15).copy()
        features_top15['importance_%'] = features_top15['importance'] * 100
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Top 15 Features")
            
            fig_features = px.bar(
                features_top15.sort_values('importance', ascending=True),
                x='importance_%',
                y='feature',
                orientation='h',
                color='importance_%',
                color_continuous_scale='Viridis',
                labels={'importance_%': 'Importance (%)', 'feature': 'Feature'}
            )
            
            fig_features.update_layout(
                height=600,
                showlegend=False,
                xaxis_title="Importance (%)",
                yaxis_title=""
            )
            
            st.plotly_chart(fig_features, use_container_width=True)
        
        with col2:
            st.subheader("Top 5 Features")
            
            top5 = features_df.head(5)
            
            for idx, row in top5.iterrows():
                st.metric(
                    label=row['feature'],
                    value=f"{row['importance']*100:.2f}%"
                )
                st.progress(row['importance'])
                st.markdown("---")
        
        # Graphique en cascade
        st.subheader("Contribution Cumulative des Features")
        
        features_cumsum = features_top15.copy()
        features_cumsum['cumulative'] = features_cumsum['importance_%'].cumsum()
        
        fig_cumul = go.Figure()
        
        fig_cumul.add_trace(go.Bar(
            x=features_cumsum['feature'],
            y=features_cumsum['importance_%'],
            name='Importance Individuelle',
            marker_color='lightblue'
        ))
        
        fig_cumul.add_trace(go.Scatter(
            x=features_cumsum['feature'],
            y=features_cumsum['cumulative'],
            name='Cumulative',
            yaxis='y2',
            marker_color='red',
            line=dict(width=3)
        ))
        
        fig_cumul.update_layout(
            height=400,
            xaxis_tickangle=-45,
            yaxis=dict(title='Importance (%)'),
            yaxis2=dict(title='Cumulative (%)', overlaying='y', side='right'),
            legend=dict(x=0.7, y=1)
        )
        
        st.plotly_chart(fig_cumul, use_container_width=True)
    
    # ==================== TAB 4: LISTE D'INTERVENTION ====================
    with tab4:
        st.header("⚠️ Liste des Étudiants Nécessitant une Intervention")
        
        # Filtrer uniquement les étudiants à risque moyen/élevé
        df_intervention = df_filtered[df_filtered['risk_level'].isin(['Moyen', 'Élevé'])].copy()
        
        # Trier par score de risque décroissant
        df_intervention = df_intervention.sort_values('risk_score', ascending=False)
        
        # Statistiques
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "🚨 Intervention Urgente",
                len(df_intervention[df_intervention['risk_score'] > 0.85])
            )
        
        with col2:
            st.metric(
                "⚠️ Risque Élevé",
                len(df_intervention[df_intervention['risk_level'] == 'Élevé'])
            )
        
        with col3:
            st.metric(
                "📊 Risque Moyen",
                len(df_intervention[df_intervention['risk_level'] == 'Moyen'])
            )
        
        st.markdown("---")
        
        # Options d'affichage
        col1, col2 = st.columns([1, 3])
        
        with col1:
            show_only_urgent = st.checkbox("Montrer uniquement urgents (>85%)", value=False)
            n_students = st.slider("Nombre d'étudiants à afficher", 10, 100, 50)
        
        if show_only_urgent:
            df_display = df_intervention[df_intervention['risk_score'] > 0.85].head(n_students)
        else:
            df_display = df_intervention.head(n_students)
        
        # Préparer le tableau
        df_table = df_display[[
            'ID_Etudiant', 'code_module', 'code_presentation',
            'risk_score', 'risk_level', 'mean_score_day180',
            'homepage', 'forumng', 'quiz', 'gender', 'age_band', 'region'
        ]].copy()
        
        df_table['risk_score'] = (df_table['risk_score'] * 100).round(2)
        df_table.columns = [
            'ID', 'Module', 'Présentation', 'Score Risque (%)', 'Niveau Risque',
            'Score Moyen', 'Homepage', 'Forum', 'Quiz', 'Genre', 'Âge', 'Région'
        ]
        
        # Fonction pour colorer les lignes
        def color_risk(row):
            if row['Score Risque (%)'] >= 85:
                return ['background-color: #ffcccc'] * len(row)
            elif row['Score Risque (%)'] >= 70:
                return ['background-color: #ffe6cc'] * len(row)
            else:
                return ['background-color: #fff9cc'] * len(row)
        
        # Afficher le tableau stylé
        st.dataframe(
            df_table.style.apply(color_risk, axis=1),
            use_container_width=True,
            height=600
        )
        
        # Bouton de téléchargement
        csv = df_table.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Télécharger la liste CSV",
            data=csv,
            file_name="intervention_list.csv",
            mime="text/csv"
        )
        
        # Scatter plot engagement vs performance
        st.subheader("Engagement vs Performance")
        
        fig_scatter = px.scatter(
            df_intervention,
            x='homepage',
            y='mean_score_day180',
            color='risk_level',
            size='risk_score',
            hover_data=['code_module', 'ID_Etudiant'],
            color_discrete_map={'Faible': '#388E3C', 'Moyen': '#FF9800', 'Élevé': '#D32F2F'},
            labels={
                'homepage': 'Activité Homepage (clics)',
                'mean_score_day180': 'Score Moyen',
                'risk_level': 'Niveau de Risque'
            }
        )
        
        fig_scatter.update_layout(height=500)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        📊 Learning Analytics Dashboard | Prédiction du Risque d'Échec Étudiant<br>
        Données: {0:,} étudiants | Modèle: Random Forest
        </div>
        """.format(len(df)),
        unsafe_allow_html=True
    )

except FileNotFoundError as e:
    st.error(f"❌ Erreur: Fichier non trouvé - {e}")
    st.info("Assurez-vous que les fichiers suivants existent:")
    st.code("""
    - reports/predictions_dev.csv
    - data/processed/final_dataset_day180.csv
    - reports/figures/feature_importance.csv
    """)

except Exception as e:
    st.error(f"❌ Erreur inattendue: {e}")
    st.exception(e)
