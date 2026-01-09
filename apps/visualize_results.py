"""
Outil pour visualiser les résultats archivés (CSV) sans relancer une comparaison.
Permet de recharger les graphiques CV vs Test et l'analyse de gap.
Organisé avec des onglets pour: Prédictions & Résidus | Analyse des Features | Comparaison des Modèles
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from sklearn.inspection import permutation_importance
from scipy.stats import chi2_contingency

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

# Fonction pour calculer Cramér's V
def cramers_v(x, y):
    """Calcule le coefficient Cramér's V pour deux variables catégoriques."""
    try:
        # Supprimer les NaN
        x_clean = x.dropna()
        y_clean = y.dropna()
        
        # Vérifier si les données nettoyées sont suffisantes
        valid_idx = x_clean.index.intersection(y_clean.index)
        if len(valid_idx) == 0:
            return 0
        
        x_clean = x_clean[valid_idx]
        y_clean = y_clean[valid_idx]
        
        confusion_matrix = pd.crosstab(x_clean, y_clean)
        if confusion_matrix.size == 0:
            return 0
        
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        min_dim = min(confusion_matrix.shape) - 1
        
        if min_dim == 0 or n == 0:
            return 0
        
        return np.sqrt(chi2 / (n * min_dim))
    except Exception as e:
        return 0

st.set_page_config(page_title="Visualiser résultats archivés", layout="wide")
st.title("Visualiser résultats archivés")
st.markdown("Charger et afficher les graphiques d'une comparaison précédente (CSV).")

# Lister les CSVs dans results/
try:
    csv_files = sorted([f.name for f in RESULTS_DIR.glob("model_comparison_*.csv")])
except Exception:
    csv_files = []

if not csv_files:
    st.warning("Aucun fichier CSV trouvé dans results/")
    st.stop()

sel_csv = st.selectbox("Choisir un fichier CSV", csv_files)
csv_path = RESULTS_DIR / sel_csv

try:
    df = pd.read_csv(csv_path)
    st.success(f"Fichier chargé: {sel_csv} ({len(df)} modèles)")
except Exception as e:
    st.error(f"Erreur lecture CSV: {e}")
    st.stop()

# Charger le JSON associé pour récupérer task, prédictions et scores par fold
json_path = RESULTS_DIR / sel_csv.replace("model_comparison_", "model_comparison_").replace(".csv", ".json")
json_data = None
task_meta = None
preds_by_model = {}
if json_path.exists():
    try:
        with open(json_path) as f:
            json_data = json.load(f)
        task_meta = json_data.get("meta", {}).get("task")
        for res in json_data.get("results", []):
            mname = res.get("model", "unknown")
            preds = res.get("_test_predictions")
            if preds:
                preds_by_model[mname] = preds
    except Exception as e:
        st.warning(f"JSON non chargé: {e}")

# Extraire les métriques (exclure _std, _params, _space, et colonnes internes)
cv_metrics = [c.replace("cv_", "") for c in df.columns 
             if c.startswith("cv_") and not c.endswith("_std") and not c.endswith("_params") and not c.endswith("_space") and not c.startswith("_")]
test_metrics = [c.replace("test_", "") for c in df.columns 
               if c.startswith("test_") and not c.endswith("_std") and not c.endswith("_params") and not c.endswith("_space") and not c.startswith("_")]
metrics_to_plot = sorted(list(set(cv_metrics) & set(test_metrics)))

if not metrics_to_plot:
    st.warning("Aucune métrique CV/Test commune trouvée.")
    st.stop()

# Rappel des métriques utilisées
st.markdown(
    """
    **Rappel des métriques :**
    - **RMSE (Root Mean Squared Error)** : erreur quadratique moyenne, penalise davantage les grosses erreurs. Plus bas est mieux.
    - **MAE (Mean Absolute Error)** : erreur absolue moyenne, robuste aux outliers. Plus bas est mieux.
    - **R2 (Coefficient de determination)** : part de variance expliquée. Entre -inf et 1. Plus haut est mieux (1 = parfait, 0 = baseline moyenne).
    """
)

st.markdown("---")

# Créer les 3 onglets principaux
tab1, tab2, tab3 = st.tabs(["📊 Prédictions & Résidus", "🔍 Analyse des Features", "📈 Comparaison des Modèles"])

# ============================================================================
# ONGLET 1 : PRÉDICTIONS & RÉSIDUS
# ============================================================================
with tab1:
    st.subheader("Prédictions & résidus (si disponibles)")
    if not preds_by_model:
        st.info("Aucune prédiction test trouvée dans le JSON. Relancez un entraînement en enregistrant les prédictions (option 'Tous les modèles' ou ajout JSON).")
    elif task_meta and task_meta != "regression":
        st.info("Visualisations de résidus disponibles uniquement pour la régression (task détectée: %s)." % task_meta)
    else:
        model_opts = list(preds_by_model.keys())
        model_selected = st.selectbox("Modèle pour les graphiques de prédiction", model_opts)
        preds = preds_by_model.get(model_selected, {})
        try:
            y_true = np.array(preds.get("y_true", []), dtype=float)
            y_pred = np.array(preds.get("y_pred", []), dtype=float)
        except Exception as e:
            y_true = np.array([])
            y_pred = np.array([])
        if y_true.size == 0:
            st.info("Pas de prédictions exploitables pour ce modèle.")
        else:
            residuals = y_true - y_pred
            df_pred = pd.DataFrame({
                "y_true": y_true,
                "y_pred": y_pred,
                "residual": residuals,
            })

            # Prédictions vs valeurs réelles avec courbe lissée
            df_sorted = df_pred.sort_values("y_pred").reset_index(drop=True)
            window_size = max(5, len(df_sorted) // 10)  # ~10% de la taille
            df_sorted["y_true_smooth"] = df_sorted["y_true"].rolling(window=window_size, center=True, min_periods=1).mean()
            
            fig_pred_curve = go.Figure()
            # Ajouter les points individuels
            fig_pred_curve.add_trace(go.Scatter(
                x=df_sorted["y_pred"], 
                y=df_sorted["y_true"],
                mode="markers",
                name="Valeurs réelles (individuelles)",
                marker=dict(size=5, opacity=0.4, color="lightblue"),
            ))
            # Ajouter la courbe lissée
            fig_pred_curve.add_trace(go.Scatter(
                x=df_sorted["y_pred"],
                y=df_sorted["y_true_smooth"],
                mode="lines",
                name="Courbe lissée",
                line=dict(color="red", width=2),
            ))
            # Ajouter la droite y=x (prédiction parfaite)
            min_ax = float(min(df_sorted["y_pred"].min(), df_sorted["y_true"].min()))
            max_ax = float(max(df_sorted["y_pred"].max(), df_sorted["y_true"].max()))
            fig_pred_curve.add_shape(type="line", x0=min_ax, y0=min_ax, x1=max_ax, y1=max_ax, 
                                    line=dict(color="gray", dash="dash", width=1))
            fig_pred_curve.update_layout(
                title="Valeurs réelles vs Prédictions avec variance",
                xaxis_title="Prédictions",
                yaxis_title="Valeurs réelles (prix)",
                hovermode="closest",
                height=500
            )
            st.plotly_chart(fig_pred_curve, width='stretch')
            st.caption("Bleu = valeurs réelles (individuelles) | Rouge = courbe lissée (variance) | Gris pointillé = prédiction parfaite")

            # Matrice de confusion "binée" (segments/quantiles)
            st.markdown("**Matrice de confusion (régression par segments)**")
            try:
                n_bins = 5  # Forcer 5 bins (quintiles)
                # Créer les bins pour y_true et y_pred
                df_pred["y_true_bin"] = pd.qcut(df_pred["y_true"], q=n_bins, duplicates="drop", labels=False)
                df_pred["y_pred_bin"] = pd.qcut(df_pred["y_pred"], q=n_bins, duplicates="drop", labels=False)
                
                # Créer une matrice de confusion
                confusion_matrix = pd.crosstab(
                    df_pred["y_true_bin"], 
                    df_pred["y_pred_bin"], 
                    rownames=["Vrai"], 
                    colnames=["Prédit"]
                )
                
                # Afficher la heatmap
                fig_confusion = go.Figure(data=go.Heatmap(
                    z=confusion_matrix.values,
                    x=[f"Segment {i+1}" for i in range(confusion_matrix.shape[1])],
                    y=[f"Segment {i+1}" for i in range(confusion_matrix.shape[0])],
                    colorscale="Blues",
                    text=confusion_matrix.values,
                    texttemplate="%{text}",
                    textfont={"size": 12},
                ))
                fig_confusion.update_layout(
                    title="Matrice de confusion (régression binée - 5 segments)",
                    xaxis_title="Prédictions (segments)",
                    yaxis_title="Valeurs réelles (segments)",
                    height=500
                )
                st.plotly_chart(fig_confusion, width='stretch')
                st.caption("Diagonale = bonnes prédictions. Hors-diagonale = erreurs. Plus la case est foncée, plus il y a d'observations.")
            except Exception as e:
                st.info(f"Impossible de créer la matrice de confusion: {e}")

            col_res1, col_res2 = st.columns(2)
            with col_res1:
                fig_res_hist = px.histogram(df_pred, x="residual", nbins=30, title="Histogramme des résidus")
                st.plotly_chart(fig_res_hist, width='stretch')
            with col_res2:
                fig_res_scatter = px.scatter(df_pred, x="y_pred", y="residual", title="Résidus vs prédictions", opacity=0.6)
                fig_res_scatter.add_hline(y=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig_res_scatter, width='stretch')

            # Erreurs par segment (quantiles de y_true)
            try:
                n_bins = 5  # Forcer 5 bins (quintiles)
                df_pred_bins = df_pred.copy()
                df_pred_bins["bin_idx"], bin_edges = pd.qcut(df_pred_bins["y_true"], q=n_bins, duplicates="drop", retbins=True)
                
                # Créer des labels informatifs avec les fourchettes de prix
                bin_labels = []
                for i in range(len(bin_edges)-1):
                    label = f"{bin_edges[i]:,.0f}€ - {bin_edges[i+1]:,.0f}€"
                    bin_labels.append(label)
                
                # Mapper les bins à des étiquettes pour y_true
                df_pred_bins["bin_label"] = pd.cut(df_pred_bins["y_true"], bins=bin_edges, labels=bin_labels, include_lowest=True)
                
                # Mapper aussi pour y_pred (pour comparer les segments)
                df_pred_bins["y_pred_bin_label"] = pd.cut(df_pred_bins["y_pred"], bins=bin_edges, labels=bin_labels, include_lowest=True)
                
                # Calculer les statistiques par segment
                agg = df_pred_bins.groupby("bin_label", observed=True).agg(
                    mae=("residual", lambda s: float(np.mean(np.abs(s)))),
                    rmse=("residual", lambda s: float(np.sqrt(np.mean(np.square(s))))),
                    count=("residual", "size"),
                ).reset_index()
                
                # Compter bonnes vs mauvaises prédictions (même segment = bon)
                good_preds = []
                bad_preds = []
                for bin_label in agg["bin_label"]:
                    mask_segment = df_pred_bins["bin_label"] == bin_label
                    segment_data = df_pred_bins[mask_segment]
                    # Bonne prédiction = même segment
                    good = (segment_data["y_pred_bin_label"] == bin_label).sum()
                    bad = (~(segment_data["y_pred_bin_label"] == bin_label)).sum()
                    good_preds.append(int(good))
                    bad_preds.append(int(bad))
                
                agg["good_preds"] = good_preds
                agg["bad_preds"] = bad_preds
                agg["accuracy_segment"] = (agg["good_preds"] / agg["count"] * 100).round(1)
                agg["percentage"] = (agg["count"] / len(df_pred_bins) * 100).round(2)
                agg["segment"] = [f"Segment {i+1}" for i in range(len(agg))]
                
                # Afficher tableau récapitulatif
                st.markdown("**Récapitulatif par segment**")
                st.caption("**MAE/RMSE en €** • Erreur moyenne du modèle sur ce segment de prix. Exemple: MAE=50k€ = le modèle prédit ±50k€ près en moyenne pour ce segment. RMSE pénalise davantage les grosses erreurs.")
                summary_table = agg[["segment", "bin_label", "count", "percentage", "good_preds", "bad_preds", "accuracy_segment", "mae", "rmse"]].copy()
                summary_table.columns = ["Segment", "Fourchette de prix", "N obs", "% du total", "Bonnes prédictions", "Erreurs", "Accuracy %", "MAE", "RMSE"]
                summary_table["MAE"] = summary_table["MAE"].map(lambda x: f"{x:,.0f}€")
                summary_table["RMSE"] = summary_table["RMSE"].map(lambda x: f"{x:,.0f}€")
                summary_table["% du total"] = summary_table["% du total"].map(lambda x: f"{x:.1f}%")
                st.dataframe(summary_table, width='stretch')
                
                # Graphique d'erreurs
                agg_melt = agg.melt(id_vars=["segment", "bin_label", "count", "percentage"], value_vars=["mae", "rmse"], var_name="metric", value_name="value")
                fig_bins = px.bar(agg_melt, x="segment", y="value", color="metric", barmode="group", 
                                 title="Erreurs par segment (quintiles)",
                                 hover_data={"bin_label": True, "count": True, "percentage": ":.1f"})
                fig_bins.update_layout(xaxis_title="Segment", yaxis_title="Erreur (€)")
                st.plotly_chart(fig_bins, width='stretch')
            except Exception as e:
                st.info(f"Impossible de calculer les segments d'erreur: {e}")

# ============================================================================
# ONGLET 2 : ANALYSE DES FEATURES
# ============================================================================
with tab2:
    st.subheader("Analyse des features — Corrélations et Importances")
    
    st.markdown("---")
    
    # Lister les modèles disponibles
    try:
        model_files = sorted([f.name for f in MODELS_DIR.glob("*.joblib")])
    except Exception:
        model_files = []

    if not model_files:
        st.info("Aucun modèle .joblib trouvé dans models/. Entraînez d'abord un modèle en enregistrant 'Tous les modèles'.")
    else:
        sel_model = st.selectbox("Sélectionner un modèle pour l'analyse des features", model_files)
        model_path = MODELS_DIR / sel_model
        
        try:
            # Charger le modèle
            pipe = joblib.load(model_path)
            st.success(f"Modèle chargé: {sel_model}")
            
            # Charger le JSON associé pour récupérer X_test et y_test
            json_path = model_path.with_suffix(".meta.json")
            if not json_path.exists():
                st.warning("Fichier meta.json non trouvé pour ce modèle.")
            else:
                with open(json_path) as f:
                    meta = json.load(f)
                
                # Récupérer le dataset et la cible
                data_path = meta.get("data")
                target_col = meta.get("target")
                test_indices = meta.get("test_indices", [])
                
                if not data_path or not target_col:
                    st.warning("Informations manquantes dans le meta.json.")
                else:
                    # Charger le dataset
                    try:
                        df_all = pd.read_csv(data_path)
                        
                        # Récupérer X_test et y_test
                        if test_indices:
                            df_test = df_all.iloc[test_indices].copy()
                        else:
                            # Fallback: utiliser les derniers 20%
                            split_idx = int(len(df_all) * 0.8)
                            df_test = df_all.iloc[split_idx:].copy()
                        
                        X_test = df_test.drop(columns=[target_col], errors="ignore")
                        y_test = df_test[target_col] if target_col in df_test.columns else None
                        
                        if y_test is None or X_test.empty:
                            st.warning("Impossible de récupérer X_test et y_test.")
                        else:
                            # === DIAGNOSTIC DES FEATURES ===
                            st.markdown("#### 🔍 Diagnostic des Features")
                            with st.expander("Voir les diagnostics (variance, redondance, anomalies)"):
                                diag_alerts = []
                                
                                # 1. Features à faible variance (< 5 uniques ou 95%+ concentration)
                                for col in X_test.columns:
                                    nunique = X_test[col].nunique()
                                    if nunique < 5:
                                        pct_top = (X_test[col].value_counts().iloc[0] / len(X_test) * 100) if nunique > 0 else 0
                                        diag_alerts.append({
                                            "Type": "⚠️ Faible variance",
                                            "Feature": col,
                                            "Détail": f"{nunique} valeurs uniques, {pct_top:.1f}% top valeur",
                                            "Action": "Considérer suppression (peu de signal)"
                                        })
                                    elif nunique > 0:
                                        pct_top = (X_test[col].value_counts().iloc[0] / len(X_test) * 100)
                                        if pct_top > 95:
                                            diag_alerts.append({
                                                "Type": "⚠️ Quasi-constant",
                                                "Feature": col,
                                                "Détail": f"{pct_top:.1f}% concentration sur 1 valeur",
                                                "Action": "Supprimer (constant)"
                                            })
                                
                                # 2. Redondances numériques (corrélation > 0.85)
                                X_test_numeric = X_test.select_dtypes(include=[np.number])
                                if X_test_numeric.shape[1] > 1:
                                    corr_matrix = X_test_numeric.corr()
                                    for i in range(len(corr_matrix.columns)):
                                        for j in range(i+1, len(corr_matrix.columns)):
                                            corr_val = abs(corr_matrix.iloc[i, j])
                                            if corr_val > 0.85:
                                                diag_alerts.append({
                                                    "Type": "🔗 Redondance",
                                                    "Feature": f"{corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}",
                                                    "Détail": f"Corrélation = {corr_val:.3f}",
                                                    "Action": "Supprimer l'une des deux"
                                                })
                                
                                # Afficher les diagnostics
                                if diag_alerts:
                                    diag_df = pd.DataFrame(diag_alerts)
                                    st.dataframe(diag_df, width='stretch')
                                else:
                                    st.success("✅ Aucun problème détecté. Features en bon état.")
                            
                            # === MATRICE DE CORRÉLATION (Redondances) ===
                            st.markdown("#### Corrélations entre features (redondances)")
                            try:
                                # Sélectionner seulement les colonnes numériques pour la corrélation
                                X_test_numeric = X_test.select_dtypes(include=[np.number])
                                
                                if X_test_numeric.shape[1] > 1:
                                    # Calculer la matrice de corrélation
                                    corr_matrix = X_test_numeric.corr()
                                    
                                    # Afficher la heatmap
                                    fig_corr = go.Figure(data=go.Heatmap(
                                        z=corr_matrix.values,
                                        x=corr_matrix.columns,
                                        y=corr_matrix.columns,
                                        colorscale="RdYlGn",
                                        zmid=0,
                                        zmin=-1,
                                        zmax=1,
                                        text=np.round(corr_matrix.values, 2),
                                        texttemplate="%{text}",
                                        textfont={"size": 8},
                                        colorbar=dict(title="Corrélation"),
                                    ))
                                    fig_corr.update_layout(
                                        title="Matrice de corrélation (features numériques)",
                                        height=600,
                                        xaxis={"side": "bottom"},
                                    )
                                    st.plotly_chart(fig_corr, width='stretch')
                                    st.caption("**Légende** • 🔴 Rouge = corrélation négative forte | 🟡 Jaune = pas de corrélation | 🟢 Vert = corrélation positive forte. Les redondances (corrélation forte en absolu) apparaissent comme des carrés verts foncés (positif) ou rouges foncés (négatif) hors diagonale.")

                                # Matrice de corrélation (features catégoriques - Cramér's V)
                                X_test_categorical = X_test.select_dtypes(include=['object', 'category', 'string'])
                                
                                # Filtrer les features catégoriques avec au moins 2 valeurs uniques et pas complètement vides
                                valid_cat_cols = [col for col in X_test_categorical.columns 
                                                if X_test_categorical[col].notna().sum() > 0 
                                                and X_test_categorical[col].nunique() > 1]
                                
                                if len(valid_cat_cols) > 1:
                                    st.subheader("Corrélations catégoriques (Cramér's V)")
                                    X_test_categorical_filtered = X_test_categorical[valid_cat_cols]
                                    
                                    # Calculer la matrice Cramér's V
                                    cramers_matrix = pd.DataFrame(
                                        [[cramers_v(X_test_categorical_filtered.iloc[:, i], X_test_categorical_filtered.iloc[:, j])
                                          for j in range(X_test_categorical_filtered.shape[1])]
                                         for i in range(X_test_categorical_filtered.shape[1])],
                                        index=X_test_categorical_filtered.columns,
                                        columns=X_test_categorical_filtered.columns
                                    )
                                    
                                    fig_cramers = go.Figure(data=go.Heatmap(
                                        x=cramers_matrix.columns,
                                        y=cramers_matrix.index,
                                        z=cramers_matrix.values,
                                        colorscale="Blues",
                                        zmin=0,
                                        zmax=1,
                                        text=np.round(cramers_matrix.values, 2),
                                        texttemplate="%{text}",
                                        textfont={"size": 8},
                                        colorbar=dict(title="Cramér's V"),
                                    ))
                                    fig_cramers.update_layout(
                                        title="Matrice de corrélation (features catégoriques)",
                                        height=600,
                                        xaxis={"side": "bottom"},
                                    )
                                    st.plotly_chart(fig_cramers, width='stretch')
                                    st.caption("**Cramér's V (0-1)** • Force d'association entre features catégoriques (pas de direction). 🔵 Bleu clair = faible association | 🔵 Bleu foncé = association forte. Les redondances apparaissent comme des carrés bleu foncé hors diagonale.")
                            except Exception as e:
                                st.warning(f"Impossible de calculer la matrice de corrélation: {e}")

                            st.markdown("#### Calcul des importances de permutation...")
                            with st.spinner("Calcul en cours (peut prendre quelques secondes)..."):
                                try:
                                    # Calculer les importances de permutation
                                    perm_importance = permutation_importance(
                                        pipe, X_test, y_test, 
                                        n_repeats=10, random_state=42, n_jobs=-1
                                    )
                                    
                                    # Créer un dataframe des importances
                                    importance_df = pd.DataFrame({
                                        "Feature": X_test.columns,
                                        "Importance": perm_importance.importances_mean,
                                        "Std": perm_importance.importances_std,
                                    }).sort_values("Importance", ascending=False)
                                    
                                    # Garder seulement les top features (éviter trop de bruit)
                                    top_n = min(15, len(importance_df))
                                    importance_df_top = importance_df.head(top_n)
                                    
                                    # Graphique des importances
                                    fig_importance = px.bar(
                                        importance_df_top,
                                        x="Importance",
                                        y="Feature",
                                        orientation="h",
                                        title="Importances des features (permutation)",
                                        labels={"Importance": "Score d'importance", "Feature": "Variable"},
                                        error_x="Std",
                                    )
                                    fig_importance.update_layout(
                                        yaxis={'categoryorder': 'total ascending'},
                                        height=500,
                                    )
                                    st.plotly_chart(fig_importance, width='stretch')
                                    
                                    # Tableau détaillé
                                    st.markdown("**Tableau détaillé (tous les features)**")
                                    display_df = importance_df.copy()
                                    display_df["Importance"] = display_df["Importance"].map(lambda x: f"{x:.6f}")
                                    display_df["Std"] = display_df["Std"].map(lambda x: f"{x:.6f}")
                                    st.dataframe(display_df, width='stretch')
                                    
                                    st.info("""
                                    **Interprétation des importances de permutation :**
                                    - **Score élevé** = le feature est important (supprimer/brouiller ses valeurs dégrade les prédictions)
                                    - **Score faible** = le feature a peu d'influence sur les prédictions
                                    - **Std** = variabilité de l'importance selon les permutations (plus faible = plus stable)
                                    - Les features avec score négatif ont peu ou pas d'influence.
                                    """)

                                    # === DIRECTION DES FEATURES (Augmente/Baisse le prix) ===
                                    st.markdown("#### Direction des features (Augmente/Baisse le prix)")
                                    try:
                                        direction_data = []
                                        
                                        # Calculer les corrélations avec la cible (y_test)
                                        for col in X_test.columns:
                                            if pd.api.types.is_numeric_dtype(X_test[col]):
                                                # Pearson pour les features numériques
                                                corr = X_test[col].corr(y_test)
                                            else:
                                                # Encoder catégorique et calculer corrélation
                                                try:
                                                    from sklearn.preprocessing import LabelEncoder
                                                    le = LabelEncoder()
                                                    encoded = le.fit_transform(X_test[col].fillna("unknown"))
                                                    corr = pd.Series(encoded).corr(y_test.reset_index(drop=True))
                                                except:
                                                    corr = 0
                                            
                                            direction_data.append({
                                                "Feature": col,
                                                "Corrélation": corr,
                                            })
                                        
                                        direction_df = pd.DataFrame(direction_data).sort_values("Corrélation", key=abs, ascending=False)
                                        
                                        # Graphique avec couleurs (rouge = baisse prix, vert = augmente prix)
                                        fig_direction = px.bar(
                                            direction_df.head(20),
                                            x="Corrélation",
                                            y="Feature",
                                            orientation="h",
                                            title="Direction des features (Impact sur le prix)",
                                            labels={"Corrélation": "Corrélation avec le prix", "Feature": "Variable"},
                                            color="Corrélation",
                                            color_continuous_scale="RdYlGn",
                                        )
                                        fig_direction.update_layout(
                                            yaxis={'categoryorder': 'total ascending'},
                                            height=500,
                                        )
                                        st.plotly_chart(fig_direction, width='stretch')
                                        
                                        st.caption("🟢 Vert = augmente le prix | 🔴 Rouge = baisse le prix | 🟡 Jaune = peu/pas d'impact")
                                        
                                    except Exception as e:
                                        st.warning(f"Impossible de calculer la direction: {e}")
                                
                                    # === EFFET D'UN FEATURE (ALE / PDP + ICE) ===
                                    st.markdown("---")
                                    st.markdown("#### Effet d'un feature (ALE / PDP + ICE)")
                                    try:
                                        num_cols = list(X_test.select_dtypes(include=[np.number]).columns)
                                        if not num_cols:
                                            st.info("Aucun feature numérique disponible pour calculer ALE/PDP.")
                                        else:
                                            col_sel1, col_sel2 = st.columns([2,1])
                                            with col_sel1:
                                                feat = st.selectbox("Sélectionner un feature numérique", num_cols)
                                            with col_sel2:
                                                method = st.radio("Méthode", ["ALE (recommandé)", "PDP + ICE"], index=0)

                                            x = X_test[feat].dropna()
                                            if x.empty:
                                                st.info("Pas de données valides pour ce feature.")
                                            else:
                                                # Option de couleur selon direction globale (corrélation avec la cible)
                                                color_by_dir = st.checkbox("Colorer la courbe selon la direction (+ vert / - rouge)", value=True)
                                                try:
                                                    dir_corr = pd.to_numeric(X_test[feat], errors="coerce").corr(pd.to_numeric(y_test, errors="coerce"))
                                                except Exception:
                                                    dir_corr = 0
                                                dir_color = "darkgreen" if (dir_corr is not None and dir_corr >= 0) else "crimson"
                                                base_line_color = dir_color if color_by_dir else "black"
                                                if method == "ALE (recommandé)":
                                                    K = st.slider("Nombre de bins (ALE)", 5, 40, 20)
                                                    qs = np.linspace(0, 1, K+1)
                                                    bins = np.unique(np.quantile(x, qs))
                                                    # Recalculer K si bins réduits
                                                    if len(bins) < 2:
                                                        st.info("Bins insuffisants pour ALE.")
                                                    else:
                                                        effects = []
                                                        mids = []
                                                        counts = []
                                                        for k in range(len(bins)-1):
                                                            low, high = bins[k], bins[k+1]
                                                            # Inclure la borne haute uniquement sur le dernier bin
                                                            if k < len(bins)-2:
                                                                mask = (X_test[feat] >= low) & (X_test[feat] < high)
                                                            else:
                                                                mask = (X_test[feat] >= low) & (X_test[feat] <= high)
                                                            idx = X_test.index[mask]
                                                            if len(idx) == 0:
                                                                effects.append(0.0)
                                                                counts.append(0)
                                                                mids.append((low+high)/2.0)
                                                                continue
                                                            X_low = X_test.loc[idx].copy()
                                                            X_high = X_low.copy()
                                                            X_low[feat] = low
                                                            X_high[feat] = high
                                                            try:
                                                                y_low = pipe.predict(X_low)
                                                                y_high = pipe.predict(X_high)
                                                            except Exception:
                                                                # fallback: no change
                                                                y_low = np.zeros(len(idx))
                                                                y_high = np.zeros(len(idx))
                                                            delta = (y_high - y_low)
                                                            effects.append(float(np.mean(delta)))
                                                            counts.append(int(len(idx)))
                                                            mids.append((low+high)/2.0)
                                                        # Accumuler et centrer
                                                        ale_vals = np.cumsum(effects)
                                                        if len(ale_vals) > 0:
                                                            ale_vals = ale_vals - np.average(ale_vals, weights=np.maximum(1, counts))
                                                        fig_ale = go.Figure()
                                                        fig_ale.add_trace(go.Scatter(x=mids, y=ale_vals, mode="lines+markers", name="ALE", line=dict(color=base_line_color)))
                                                        fig_ale.add_hline(y=0, line_dash="dash", line_color="gray")
                                                        fig_ale.update_layout(title=f"ALE 1D — Effet de {feat}", xaxis_title=feat, yaxis_title="Effet accumulé (Δ€)", height=450)
                                                        st.plotly_chart(fig_ale, width='stretch')
                                                        with st.expander("📖 Comment lire l'ALE ?"):
                                                            st.markdown(f"""
**Axe Y (Effet accumulé en Δ€)** : Impact en euros sur la prédiction du prix.
- **Pente positive** ↗️ : Quand {feat} augmente → prix augmente (+impact).
- **Pente négative** ↘️ : Quand {feat} augmente → prix diminue (-impact).
- **Pente plate** → : Changement de {feat} n'a peu/pas d'impact sur le prix.

**Lecture concrète** :
- Si la courbe monte de 50k€ entre gauche et droite : le feature peut impacter le prix de ±50k€.
- La hauteur absolue n'a pas d'importance (elle est centrée à 0) → regardez les **variations/pentes**.
- Plus la pente est **raide**, plus le feature est sensible (petit changement → grand impact €).

**Exemple** : Surface de 80m² → +200k€, 150m² → +300k€ = pente positive (plus grand = plus cher).

**Robustesse** : ALE gère les corrélations entre features (PDP a des biais si X1 corrélé à X2).
                                                            """)
                                                else:
                                                    n_grid = st.slider("Points du grid (PDP)", 10, 50, 20)
                                                    n_ice = st.slider("Courbes ICE (échantillon)", 20, 300, 100)
                                                    grid = np.quantile(x, np.linspace(0, 1, n_grid))
                                                    grid = np.unique(grid)
                                                    rng = np.random.RandomState(42)
                                                    idx_sample = rng.choice(X_test.index, size=min(n_ice, len(X_test)), replace=False)
                                                    ice_traces = []
                                                    preds_matrix = []
                                                    for i in idx_sample:
                                                        row = X_test.loc[[i]].copy()
                                                        X_rep = pd.concat([row]*len(grid), ignore_index=True)
                                                        X_rep[feat] = grid
                                                        try:
                                                            y_pred_line = pipe.predict(X_rep)
                                                        except Exception:
                                                            y_pred_line = np.zeros(len(grid))
                                                        preds_matrix.append(y_pred_line)
                                                        ice_traces.append(dict(x=grid, y=y_pred_line))
                                                    preds_matrix = np.array(preds_matrix)
                                                    pdp = preds_matrix.mean(axis=0) if preds_matrix.size else np.zeros(len(grid))
                                                    # Plot
                                                    fig_pdp = go.Figure()
                                                    # ICE lines
                                                    for tr in ice_traces:
                                                        fig_pdp.add_trace(go.Scatter(x=tr['x'], y=tr['y'], mode="lines", line=dict(color="rgba(0,0,0,0.15)"), showlegend=False))
                                                    # PDP line
                                                    fig_pdp.add_trace(go.Scatter(x=grid, y=pdp, mode="lines", name="PDP", line=dict(color=base_line_color, width=3)))
                                                    fig_pdp.update_layout(title=f"PDP + ICE — Effet de {feat}", xaxis_title=feat, yaxis_title="Prédiction (€)", height=450)
                                                    st.plotly_chart(fig_pdp, width='stretch')
                                                    st.caption("PDP: effet moyen; ICE: effet par observation (peut révéler des interactions).")
                                    except Exception as e:
                                        st.warning(f"Impossible de calculer ALE/PDP: {e}")
                                
                                except Exception as e:
                                    st.error(f"Erreur calcul importances: {e}")
                    except Exception as e:
                        st.error(f"Erreur chargement dataset: {e}")
        except Exception as e:
            st.error(f"Erreur chargement modèle: {e}")

# ============================================================================
# ONGLET 3 : COMPARAISON DES MODÈLES
# ============================================================================
with tab3:
    st.subheader("Comparaison des performances CV vs Test")
    
    # Afficher les métadonnées de la comparaison
    try:
        if json_data and "meta" in json_data:
            meta = json_data["meta"]
            cv_folds = meta.get("cv_folds", "?")
            test_size = meta.get("test_size", "?")
            created_at = meta.get("created_at", "?")
            n_metrics = len(metrics_to_plot)
            
            col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)
            with col_meta1:
                st.metric("CV Folds", cv_folds)
            with col_meta2:
                st.metric("Test Size", f"{test_size*100:.0f}%" if isinstance(test_size, float) else test_size)
            with col_meta3:
                st.metric("Métriques", n_metrics)
            with col_meta4:
                st.metric("Créé le", created_at[:10] if isinstance(created_at, str) else created_at)
    except Exception:
        pass
    
    st.markdown("---")
    
    # Tableau récapitulatif des modèles et métriques
    st.markdown("**Tableau récapitulatif — Modèles, métriques et diagnostics**")
    try:
        # Utilitaire: déterminer si plus haut est mieux pour une métrique
        def _higher_is_better(metric_name: str) -> bool:
            m = metric_name.lower()
            return m in ("r2", "accuracy", "f1", "roc_auc")

        # KPIs management: meilleur modèle par métrique (basé sur Test)
        kpi_cols = st.columns(max(1, min(4, len(metrics_to_plot))))
        for kpi_idx, metric in enumerate(metrics_to_plot[:len(kpi_cols)]):
            cv_col = f"cv_{metric}"
            test_col = f"test_{metric}"
            valid = df[["model", test_col, cv_col]].dropna()
            if not valid.empty:
                hib = _higher_is_better(metric)
                best_row = valid.loc[valid[test_col].idxmax()] if hib else valid.loc[valid[test_col].idxmin()]
                best_model = str(best_row["model"])
                best_test = float(best_row[test_col])
                best_cv = float(best_row[cv_col]) if cv_col in best_row.index and pd.notna(best_row[cv_col]) else None
                delta = None if best_cv is None else (best_test - best_cv)
                with kpi_cols[kpi_idx]:
                    st.metric(label=f"Meilleur {metric.upper()} — {best_model}", value=round(best_test, 4), delta=(round(delta, 4) if delta is not None else None))

        # Détails des KPI (CV reconstitué, écarts-types et folds)
        details = []
        for metric in metrics_to_plot:
            cv_col = f"cv_{metric}"
            test_col = f"test_{metric}"
            cv_std_col = f"cv_{metric}_std"
            test_std_col = f"test_{metric}_std"
            if cv_col in df.columns and test_col in df.columns:
                hib = _higher_is_better(metric)
                valid = df[["model", test_col, cv_col, cv_std_col, test_std_col]].copy()
                # Handle missing std columns
                if cv_std_col not in valid.columns:
                    valid[cv_std_col] = np.nan
                if test_std_col not in valid.columns:
                    valid[test_std_col] = np.nan
                valid = valid.dropna(subset=[test_col, cv_col])
                if not valid.empty:
                    idx = valid[test_col].idxmax() if hib else valid[test_col].idxmin()
                    row = valid.loc[idx]
                    cv_folds_val = json_data.get("meta", {}).get("cv_folds") if json_data else None
                    details.append({
                        "Métrique": metric.upper(),
                        "Modèle": row["model"],
                        "Test": round(float(row[test_col]), 4),
                        "CV": round(float(row[cv_col]), 4),
                        "Delta (Test - CV)": round(float(row[test_col] - row[cv_col]), 4),
                        "CV std": (round(float(row[cv_std_col]), 4) if pd.notna(row[cv_std_col]) else None),
                        "Test std": (round(float(row[test_std_col]), 4) if pd.notna(row[test_std_col]) else None),
                        "CV folds": cv_folds_val,
                    })
        if details:
            st.markdown("**Détails des KPI (CV/Test, Δ, std, folds)**")
            st.dataframe(pd.DataFrame(details), use_container_width=True, hide_index=True)
        summary_rows = []
        for idx, row_data in df.iterrows():
            model_name = row_data.get("model", "unknown")
            for metric in metrics_to_plot:
                cv_col = f"cv_{metric}"
                test_col = f"test_{metric}"
                cv_std_col = f"cv_{metric}_std"
                test_std_col = f"test_{metric}_std"
                cv_val = pd.to_numeric(row_data.get(cv_col), errors='coerce') if cv_col in row_data.index else np.nan
                test_val = pd.to_numeric(row_data.get(test_col), errors='coerce') if test_col in row_data.index else np.nan
                cv_std = pd.to_numeric(row_data.get(cv_std_col), errors='coerce') if cv_std_col in row_data.index else np.nan
                test_std = pd.to_numeric(row_data.get(test_std_col), errors='coerce') if test_std_col in row_data.index else np.nan
                gap = cv_val - test_val if pd.notna(cv_val) and pd.notna(test_val) else np.nan
                gap_pct = (gap / cv_val * 100) if pd.notna(gap) and cv_val not in (0, None) else np.nan
                # Formatage CV ± Std et Test ± Std
                cv_str = f"{cv_val:.4f} ± {cv_std:.4f}" if pd.notna(cv_val) and pd.notna(cv_std) else (f"{cv_val:.4f}" if pd.notna(cv_val) else "N/A")
                test_str = f"{test_val:.4f} ± {test_std:.4f}" if pd.notna(test_val) and pd.notna(test_std) else (f"{test_val:.4f}" if pd.notna(test_val) else "N/A")
                # Diagnostic simple
                if pd.isna(gap_pct):
                    diag = "N/A"
                else:
                    tol = 2.0  # seuil (%) pour considérer 'stable'
                    if abs(gap_pct) <= tol:
                        diag = "Stable"
                    elif gap_pct > 0:
                        diag = "Sous-apprentissage (CV > Test)"
                    else:
                        diag = "Sur-apprentissage (CV < Test)"
                summary_rows.append({
                    "Modèle": model_name,
                    "Métrique": metric.upper(),
                    "CV (mean ± std)": cv_str,
                    "Test (mean ± std)": test_str,
                    "Gap (CV-Test)": float(gap) if pd.notna(gap) else None,
                    "Gap (%)": float(gap_pct) if pd.notna(gap_pct) else None,
                    "Diagnostic": diag,
                })
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            # Formatage léger pour Gap
            def fmt(x):
                return None if pd.isna(x) else round(float(x), 4)
            for col in ["Gap (CV-Test)", "Gap (%)"]:
                if col in summary_df.columns:
                    summary_df[col] = summary_df[col].map(fmt)
            # Diagnostics colorés (management-friendly)
            def diag_color(d: str) -> str:
                if d.startswith("Stable"):
                    return "Stable 🟢"
                if d.startswith("Sous-apprentissage"):
                    return "Sous-apprentissage 🟠"
                if d.startswith("Sur-apprentissage"):
                    return "Sur-apprentissage 🔴"
                return d
            summary_df["Diagnostic"] = summary_df["Diagnostic"].map(diag_color)

            # Affichage direct des données (sans filtre ni tri)
            df_display = summary_df.copy()
            
            # Trois tableaux côte à côte pour comparer les modèles
            models_to_show = list(df_display["Modèle"].unique())
            
            if len(models_to_show) > 0:
                # Créer jusqu'à 3 colonnes (un tableau par modèle)
                cols = st.columns(min(3, len(models_to_show)))
                
                for col_idx, (col, model_name) in enumerate(zip(cols, models_to_show[:3])):
                    with col:
                        # Filtrer les données pour ce modèle
                        model_data = df_display[df_display["Modèle"] == model_name].copy()
                        
                        # Sélectionner les colonnes à afficher (Diagnostic en première colonne)
                        cols_to_show = ["Diagnostic", "Métrique", "CV (mean ± std)", "Test (mean ± std)", "Gap (CV-Test)", "Gap (%)"]
                        model_data = model_data[[c for c in cols_to_show if c in model_data.columns]]
                        
                        # Afficher le titre du modèle
                        st.markdown(f"### {model_name}")
                        
                        # Afficher le tableau
                        st.dataframe(model_data, use_container_width=True, hide_index=True)
            
            
            # Heatmap de performance relative (vert = meilleur)
            st.markdown("---")
            st.markdown("### 🎨 Carte de Performance Relative (Normalized)")
            st.markdown("""
            **À quoi ça sert ?** Comparer visuellement tous les modèles sur toutes les métriques en un coup d'œil.
            
            **Comment ça marche :**
            - Chaque cellule est **normalisée entre 0 et 1**
            - **1.0 (vert foncé)** = meilleur score sur cette métrique
            - **0.5 (vert clair)** = score moyen
            - **0.0 (blanc)** = pire score sur cette métrique
            - Les chiffres dans les cellules montrent la valeur normalisée (ex. 0.95, 0.52, 0.10)
            
            **Lecture rapide :** Regardez les **lignes** (modèles) — celle avec le plus de vert foncé est la meilleure overall !
            """)
            try:
                models = df["model"].tolist()
                perf_mat = []
                for metric in metrics_to_plot:
                    test_col = f"test_{metric}"
                    col_vals = pd.to_numeric(df[test_col], errors='coerce') if test_col in df.columns else pd.Series([np.nan]*len(df))
                    vals = col_vals.values.astype(float)
                    vmin = np.nanmin(vals)
                    vmax = np.nanmax(vals)
                    if np.isnan(vmin) or np.isnan(vmax) or vmax == vmin:
                        norm = np.zeros_like(vals)
                    else:
                        norm = (vals - vmin) / (vmax - vmin)
                    if not _higher_is_better(metric):
                        norm = 1.0 - norm
                    perf_mat.append(norm)
                perf_arr = np.vstack(perf_mat).T
                fig_perf = go.Figure(data=go.Heatmap(
                    z=perf_arr,
                    x=[m.upper() for m in metrics_to_plot],
                    y=models,
                    colorscale="Greens",
                    zmin=0,
                    zmax=1,
                    text=np.round(perf_arr, 2),
                    texttemplate="%{text}",
                    textfont={"size": 8},
                    colorbar=dict(title="Performance relative"),
                ))
                fig_perf.update_layout(height=400, xaxis={"side": "bottom"})
                st.plotly_chart(fig_perf, width='stretch')
                
                # Explication détaillée
                with st.expander("📖 Comment lire cette carte ?"):
                    st.markdown("""
                    **Exemple concret :**
                    
                    Supposons tes modèles et métriques :
                    | Modèle | MAE | RMSE | R2 |
                    |--------|-----|------|-----|
                    | Random Forest | 8461 (meilleur) | 42229 | 0.9973 (meilleur) |
                    | Gradient Boosting | 9200 | 45000 (meilleur) | 0.9950 |
                    | Linear | 12000 (pire) | 65000 (pire) | 0.9800 (pire) |
                    
                    **La heatmap après normalisation :**
                    - Random Forest / MAE → 1.0 ✅ (vert foncé)
                    - Gradient Boosting / RMSE → 1.0 ✅ (vert foncé)
                    - Linear / tout → ~0.0 ❌ (blanc)
                    
                    **Interprétation :**
                    - Random Forest a 2 cellules vertes foncées → **winner overall** 🏆
                    - Gradient Boosting excelle sur RMSE mais pas ailleurs
                    - Linear est mauvais partout → à éliminer
                    
                    **Cas particuliers :**
                    - Tous les modèles égaux sur une métrique ? → colonne grise (uniforme)
                    - Métrique trop facile/instable ? → parfois une cellule isolée très verte
                    """)
                
                st.caption("💡 **Conseil :** Plus de vert = modèle fiable. Cherchez le modèle avec la ligne la plus verte.")
            except Exception as e:
                st.warning(f"Impossible d'afficher la carte de performance: {e}")
        else:
            st.info("Aucune donnée pour construire le récapitulatif.")
    except Exception as e:
        st.warning(f"Impossible de générer le tableau récapitulatif: {e}")
    
    # Plot 1: CV vs Test pour chaque métrique
    fig = make_subplots(
        rows=1, cols=len(metrics_to_plot),
        subplot_titles=[f"{m.upper()}" for m in metrics_to_plot],
        specs=[[{"secondary_y": False} for _ in metrics_to_plot]]
    )

    for col_idx, metric in enumerate(metrics_to_plot, start=1):
        cv_col = f"cv_{metric}"
        test_col = f"test_{metric}"
        cv_std_col = f"cv_{metric}_std"
        test_std_col = f"test_{metric}_std"
        
        if cv_col in df.columns and test_col in df.columns:
            cv_vals = pd.to_numeric(df[cv_col], errors='coerce')
            test_vals = pd.to_numeric(df[test_col], errors='coerce')
            cv_std = pd.to_numeric(df[cv_std_col], errors='coerce') if cv_std_col in df.columns else None
            test_std = pd.to_numeric(df[test_std_col], errors='coerce') if test_std_col in df.columns else None
            
            fig.add_trace(
                go.Bar(
                    x=df["model"],
                    y=cv_vals,
                    name=f"{metric} (CV)",
                    marker_color="lightblue",
                    error_y=dict(type="data", array=cv_std, color="red") if cv_std is not None else None,
                ),
                row=1, col=col_idx
            )
            fig.add_trace(
                go.Bar(
                    x=df["model"],
                    y=test_vals,
                    name=f"{metric} (Test)",
                    marker_color="coral",
                    error_y=dict(type="data", array=test_std, color="red") if test_std is not None else None,
                ),
                row=1, col=col_idx
            )

    fig.update_layout(height=400, showlegend=True, title_text="Comparaison CV vs Test par métrique")
    st.plotly_chart(fig, width='stretch')

    st.markdown("""
    **Comment lire le graphique CV vs Test :**
    - Barres bleues = CV (validation croisee), barres corail = Test final.
    - RMSE/MAE : plus bas est mieux. R2 : plus haut est mieux.
    - CV proche du Test indique une generalisation stable. Un Test nettement plus mauvais que le CV suggere biais/sous-apprentissage; l'inverse peut indiquer variance/chance.
    """)

    # Plot 1b: Line plot des scores CV (variance par fold)
    st.markdown("**Variance des scores CV (évolution par fold)**")
    cv_fold_data = []
    if json_data:
        try:
            results = json_data.get("results", [])
            for res in results:
                model_name = res.get("model", "unknown")
                fold_scores = res.get("_cv_scores_per_fold", [])
                if fold_scores:
                    for metric in metrics_to_plot:
                        for fold_idx, fold_dict in enumerate(fold_scores):
                            if metric in fold_dict:
                                cv_fold_data.append({
                                    "Model": model_name,
                                    "Métrique": metric,
                                    "Fold": fold_idx + 1,
                                    "Score": fold_dict[metric],
                                })
        except Exception as e:
            st.warning(f"JSON non chargé ou sans _cv_scores_per_fold: {e}")

    if cv_fold_data:
        cv_fold_df = pd.DataFrame(cv_fold_data)
        fig_line = px.line(cv_fold_df, x="Fold", y="Score", color="Model", facet_col="Métrique", facet_col_wrap=3,
                           title="Variance des scores CV par fold (Line plot)",
                           markers=True)
        fig_line.update_yaxes(rangemode="tozero")
        st.plotly_chart(fig_line, width='stretch')
        st.markdown("""
        **Comment lire le graphique:**
        - Ligne plate = modèle stable (variance faible). Ligne oscillante = instabilité (variance élevée).
        - Pente ascendante/descendante = biais systématique sur certains splits.
        - Chaque point = score d'un fold de validation croisée.
        """)
    else:
        st.info("Aucun fichier JSON trouvé ou _cv_scores_per_fold absent. Relancez une comparaison pour avoir les scores par fold.")

    st.markdown("---")

    # Plot 2: Generalization Gap (CV - Test)
    st.markdown("**Généralization Gap**")
    display_mode = st.radio("Affichage du gap", ["Valeur brute", "Pourcentage du CV"], index=0, horizontal=True)
    gap_data = []
    for idx, row_data in df.iterrows():
        for metric in metrics_to_plot:
            cv_col = f"cv_{metric}"
            test_col = f"test_{metric}"
            if cv_col in row_data.index and test_col in row_data.index:
                try:
                    cv_val = float(row_data[cv_col])
                    test_val = float(row_data[test_col])
                    gap = cv_val - test_val
                    pct = (gap / cv_val * 100) if cv_val not in (0, None) else None
                    gap_data.append({
                        "Model": row_data["model"],
                        "Métrique": metric,
                        "Gap (CV-Test)": gap,
                        "Gap (%)": pct,
                    })
                except:
                    pass

    if gap_data:
        gap_df = pd.DataFrame(gap_data)
        y_col = "Gap (CV-Test)" if display_mode == "Valeur brute" else "Gap (%)"
        fig_gap = px.bar(
            gap_df,
            x="Model",
            y=y_col,
            color="Métrique",
            title="Écart CV - Test (négatif=overfitting, positif=underfitting)",
            barmode="group",
        )
        fig_gap.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_gap, width='stretch')
        
        st.markdown("""
        **Interprétation:**
        - **Gap ≈ 0**: Bonne generalisation (CV ≈ Test)
        - **Gap > 0 (CV > Test)**: Possible sous-apprentissage/biais (le Test degrade)
        - **Gap < 0 (CV < Test)**: Possible sur-apprentissage/variance (le Test semble meilleur que CV)
        - Mode *Pourcentage du CV* : le gap est exprime en % du score CV pour rendre comparables les ordres de grandeur (utile pour R2 vs RMSE/MAE).
        """)

    # Tableau détaillé
    st.markdown("---")
    st.markdown("**Tableau détaillé — Tous les modèles**")
    st.dataframe(df, width='stretch')
