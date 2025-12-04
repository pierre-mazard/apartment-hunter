"""Application Streamlit pour comparer interactivement deux jeux de données
et justifier un choix de jeu pour l'entraînement.

Exécution:
    streamlit run apps/compare_datasets_streamlit.py

Résumé des fonctionnalités:
- Chargement des CSV dans `data/`
- Indicateurs synthétiques orientés management (KPIs, couleurs, emoji)
- Préréglages pour pondérations + bouton d'application
- Visualisations compactes (missingness, distribution, corrélations)
- Tableaux détaillés accessibles via expanders pour éviter la surcharge
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.express as px
import streamlit.components.v1 as components
from pathlib import Path
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
from textwrap import dedent
import json
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def summarize(df: pd.DataFrame) -> dict:
    numeric = df.select_dtypes(include='number')
    categorical = df.select_dtypes(include=['object', 'category'])
    # estimate fraction of numeric cells that are outliers (IQR method)
    def _outlier_frac(numeric_df: pd.DataFrame) -> float:
        if numeric_df.shape[1] == 0:
            return 0.0
        non_null_cells = int(numeric_df.count().sum())
        if non_null_cells == 0:
            return 0.0
        outlier_cells = 0
        for col in numeric_df.columns:
            ser = numeric_df[col].dropna()
            if ser.empty:
                continue
            q1 = ser.quantile(0.25)
            q3 = ser.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_cells += int(((numeric_df[col] < lower) | (numeric_df[col] > upper)).sum())
        return float(outlier_cells) / non_null_cells * 100.0
    return {
        'rows': len(df),
        'cols': df.shape[1],
        'num_numeric': numeric.shape[1],
        'num_categorical': categorical.shape[1],
        'missing_total': int(df.isna().sum().sum()),
        'missing_perc': float(df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100),
        'outlier_perc': _outlier_frac(numeric),
    }


def detect_target(df: pd.DataFrame) -> str | None:
    # Improved heuristic: prefer explicit candidates, but when multiple
    # "price-like" columns exist, pick the one most correlated (abs)
    # with other numeric features.
    tokens = ['price', 'prix', 'precio', 'sale', 'buy', 'purchase', 'rent']
    # gather columns whose name contains any token
    price_like = [c for c in df.columns if any(t in c.lower() for t in tokens)]
    # if none found, fallback to older simple scan for 'price' substrings
    if not price_like:
        for c in df.select_dtypes(include='number').columns:
            if 'price' in c.lower() or 'precio' in c.lower():
                return c
        return None

    # if only one candidate, return it
    if len(price_like) == 1:
        return price_like[0]

    # attempt correlation-based selection among numeric candidates
    numeric = df.select_dtypes(include='number')
    if numeric.shape[1] >= 2:
        try:
            corr = numeric.corr().abs()
            best_col = None
            best_score = -1.0
            for col in price_like:
                if col not in corr.columns:
                    continue
                # average absolute correlation with other numeric cols
                col_corr = corr[col].drop(labels=[col], errors='ignore')
                if col_corr.empty:
                    score = 0.0
                else:
                    score = float(col_corr.mean(skipna=True))
                if score > best_score:
                    best_score = score
                    best_col = col
            if best_col:
                return best_col
        except Exception:
            # if correlation computation fails, fall through to priority rule
            pass

    # fallback priority: prefer sale/buy/purchase, then price, then rent
    priority = ['sale', 'buy', 'purchase', 'price', 'precio', 'rent']
    for p in priority:
        for c in price_like:
            if p in c.lower():
                return c
    # otherwise return the first match
    return price_like[0]


def compute_informativeness(df: pd.DataFrame) -> dict:
    info: dict = {}
    info['rows'] = len(df)
    info['cols'] = df.shape[1]
    info['target'] = detect_target(df)
    info['num_numeric'] = df.select_dtypes(include='number').shape[1]
    info['num_categorical'] = df.select_dtypes(include='object').shape[1]
    info['missing_total'] = int(df.isna().sum().sum())
    info['missing_perc'] = float(df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
    info['geo'] = any(c.lower() in ('lat', 'latitude', 'lon', 'lng', 'longitude') for c in df.columns)
    numeric = df.select_dtypes(include='number')
    # per-column outlier percentages (IQR)
    outlier_stats = []
    for col in numeric.columns:
        ser = numeric[col].dropna()
        if ser.empty:
            continue
        q1 = ser.quantile(0.25)
        q3 = ser.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            out_pct = 0.0
            out_cnt = 0
        else:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            out_mask = (numeric[col] < lower) | (numeric[col] > upper)
            out_cnt = int(out_mask.sum())
            non_null = int(numeric[col].count())
            out_pct = (out_cnt / non_null * 100.0) if non_null > 0 else 0.0
        outlier_stats.append((col, out_cnt, out_pct))
    outlier_stats.sort(key=lambda x: x[2], reverse=True)
    info['outlier_on_top'] = outlier_stats[:20]
    # aggregate outlier percent across numeric cells
    total_non_null = int(numeric.count().sum())
    total_out = sum([c for (_, c, _) in outlier_stats])
    info['outlier_perc'] = float(total_out / total_non_null * 100.0) if total_non_null > 0 else 0.0
    preds = [c for c in numeric.columns if c.lower() not in ('id', 'index')]
    info['top_preds_sample'] = preds[:20]
    rows = len(df)
    missing_on_top = [(c, int(df[c].isna().sum()), df[c].isna().sum() / rows * 100) for c in info['top_preds_sample']]
    info['missing_on_top'] = sorted(missing_on_top, key=lambda x: x[2])
    if info['target'] and info['target'] in numeric.columns:
        corr = numeric.corr()[info['target']].abs().sort_values(ascending=False).drop(labels=[info['target']], errors='ignore')
        info['top_corr'] = corr.head(10).to_dict()
    else:
        info['top_corr'] = {}
    if info['target'] and info['target'] in numeric.columns and len(preds) > 0:
        X = numeric.drop(columns=[info['target']]) if info['target'] in numeric.columns else numeric
        X = X.fillna(0)
        y = numeric[info['target']].fillna(0)
        try:
            mi = mutual_info_regression(X, y, random_state=0)
            mi_s = pd.Series(mi, index=X.columns).sort_values(ascending=False).head(10)
            info['top_mi'] = mi_s.to_dict()
        except Exception:
            info['top_mi'] = {}
    else:
        info['top_mi'] = {}
    return info


def main() -> None:
    st.set_page_config(page_title='Comparaison datasets', layout='wide')
    st.title('Comparaison interactive des jeux de données')

    a_path = DATA_DIR / 'houses_Madrid.csv'
    b_path = DATA_DIR / 'kc_house_data.csv'

    # default load original datasets
    df_a = load_csv(a_path) if a_path.exists() else pd.DataFrame()
    df_b = load_csv(b_path) if b_path.exists() else pd.DataFrame()

    info_a = summarize(df_a)
    info_b = summarize(df_b)

    # KPI cards
    def render_kpis(title: str, info: dict, df: pd.DataFrame | None = None, path: Path | None = None):
        missing_pct = info['missing_perc']
        outlier_pct = info.get('outlier_perc', 0.0)
        # prefer explicit df for target detection when available
        try:
            if df is not None:
                tgt = detect_target(df)
            elif path is not None:
                tgt = detect_target(load_csv(path))
            else:
                tgt = None
        except Exception:
            tgt = None
        c1, c2, c3 = st.columns([1, 1, 1])
        c1.metric(label=f"{title} — Lignes", value=f"{info['rows']:,}")
        c2.metric(label=f"{title} — Colonnes", value=f"{info['cols']}")
        if missing_pct < 5:
            color = '#16a34a'; emoji = '✅'
        elif missing_pct < 25:
            color = '#f59e0b'; emoji = '⚠️'
        else:
            color = '#dc2626'; emoji = '❌'
        c3.markdown(f"<div style='font-size:15px'>{emoji} Valeurs manquantes: <b style='color:{color}'>{missing_pct:.2f}%</b></div>", unsafe_allow_html=True)
        st.markdown(f"**Cible détectée:** {tgt}")
        st.markdown(f"**% Aberrants (numeric):** {outlier_pct:.2f}%")

    # Ensure cleaned versions and pipeline exist in session state
    if 'df_a_clean' not in st.session_state:
        st.session_state['df_a_clean'] = df_a.copy()
    if 'df_b_clean' not in st.session_state:
        st.session_state['df_b_clean'] = df_b.copy()
    if 'pipeline_a' not in st.session_state:
        st.session_state['pipeline_a'] = []
    if 'pipeline_b' not in st.session_state:
        st.session_state['pipeline_b'] = []
    # optional custom datasets loaded by user in the 'Chargement' tab
    if 'df_a_custom' in st.session_state:
        df_a = st.session_state['df_a_custom']
    if 'df_b_custom' in st.session_state:
        df_b = st.session_state['df_b_custom']
    # display names for datasets (original filename or uploaded name)
    name_a = st.session_state.get('name_a', a_path.name)
    name_b = st.session_state.get('name_b', b_path.name)

    # Sidebar: presets + sliders (moved here so it's available across tabs)
    st.sidebar.header('Comparaison & pondérations')
    st.sidebar.markdown('Sélectionnez un *préréglage* pour appliquer rapidement une configuration de poids, ou personnalisez ci‑dessous.')
    presets = {
        'Prototype rapide': (3.0, 1.0, 1.0, 1.0),
        'Qualité robuste': (2.0, 1.0, 4.0, 2.0),
        'Favoriser quantité': (1.0, 3.0, 1.0, 0.5),
        'Minimal nettoyage': (1.0, 1.0, 0.5, 0.5),
    }
    preset = st.sidebar.selectbox('Préréglages', list(presets.keys()), index=0)
    w_t_def, w_r_def, w_m_def, w_o_def = presets[preset]
    # callback to apply preset values into session_state safely
    def _apply_preset_cb(w_t: float, w_r: float, w_m: float, w_o: float) -> None:
        st.session_state['w_target'] = float(w_t)
        st.session_state['w_rows'] = float(w_r)
        st.session_state['w_missing'] = float(w_m)
        st.session_state['w_outlier'] = float(w_o)
    # place the apply button right under the presets selectbox (sidebar)
    st.sidebar.button('Appliquer le préréglage', on_click=_apply_preset_cb, kwargs={'w_t': w_t_def, 'w_r': w_r_def, 'w_m': w_m_def, 'w_o': w_o_def})
    st.sidebar.markdown('')
    # use session_state values if present so callbacks can update sliders
    w_target = st.sidebar.slider('Poids — présence colonne cible', 0.0, 5.0, value=st.session_state.get('w_target', float(w_t_def)), key='w_target')
    w_rows = st.sidebar.slider('Poids — taille (n lignes)', 0.0, 5.0, value=st.session_state.get('w_rows', float(w_r_def)), key='w_rows')
    w_missing = st.sidebar.slider('Poids — qualité (moins de valeurs manquantes)', 0.0, 5.0, value=st.session_state.get('w_missing', float(w_m_def)), key='w_missing')
    w_outlier = st.sidebar.slider("Poids — proportion de valeurs aberrantes (numériques)", 0.0, 5.0, value=st.session_state.get('w_outlier', float(w_o_def)), key='w_outlier')
    with st.sidebar.expander('Aide — signification des poids'):
        st.markdown(dedent('''
        • **Présence colonne cible** — avantage si une colonne cible (prix) est déjà présente, réduit le besoin d'étiquetage manuel.
        • **Taille (lignes)** — favorise les jeux avec plus de lignes pour une meilleure généralisation.
        • **Qualité (missing)** — pénalise les jeux contenant beaucoup de valeurs manquantes.
        • **Valeurs aberrantes** — pénalise les jeux où la proportion de valeurs numériques anormales (IQR) est élevée.

        *Conseil visuel* : choisissez *Prototype rapide* pour itérations rapides, *Qualité robuste* pour un jeu plus fiable en production.
        '''))
    palette = st.sidebar.selectbox('Palette de couleurs', ['Vert / Rouge', 'Bleu / Orange'])

    # Top-level tabs: Chargement, Diagnostic, Exploration, Nettoyage, Comparaison
    tab_load, tab_diag, tab_explore, tab_clean, tab_compare = st.tabs(['Chargement — Jeux de données', 'Diagnostic — Qualité', 'Exploration des données', 'Nettoyage — Préparer', 'Comparaison'])

    # helper: apply pipeline of operations (simple replayable ops)
    def apply_pipeline(original: pd.DataFrame, pipeline: list[dict]) -> pd.DataFrame:
        df = original.copy()
        for op in pipeline:
            if op.get('op') == 'drop':
                cols = op.get('cols', [])
                df = df.drop(columns=cols, errors='ignore')
            elif op.get('op') == 'drop_rows':
                # drop rows by explicit index list
                idxs = op.get('index', []) or []
                try:
                    df = df.drop(index=[i for i in idxs if i in df.index], errors='ignore')
                except Exception:
                    try:
                        # try positional indices
                        df = df.drop(index=idxs, errors='ignore')
                    except Exception:
                        continue
            elif op.get('op') == 'drop_rows_cond':
                # drop rows matching a pandas query condition stored as string
                cond = op.get('condition')
                if not cond:
                    continue
                try:
                    mask = df.query(cond).index
                    df = df.drop(index=mask, errors='ignore')
                except Exception:
                    # if query fails, skip
                    continue
            elif op.get('op') == 'drop_rows_where_na':
                col = op.get('col')
                if not col:
                    continue
                if col not in df.columns:
                    continue
                try:
                    mask = df[col].isna()
                    df = df.loc[~mask]
                except Exception:
                    continue
            elif op.get('op') == 'impute':
                col = op.get('col')
                val = op.get('val')
                if col in df.columns:
                    df[col] = df[col].fillna(val)
            elif op.get('op') == 'cast':
                cols = op.get('cols', [])
                dtype = op.get('dtype')
                for col in cols:
                    if col not in df.columns:
                        continue
                    try:
                        if dtype == 'int':
                            # use pandas nullable integer to preserve NA
                            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                        elif dtype == 'float':
                            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
                        elif dtype == 'datetime':
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                        elif dtype == 'bool':
                            try:
                                s = df[col]
                                if pd.api.types.is_numeric_dtype(s):
                                    df[col] = s.fillna(0).astype(int).astype(bool).astype('boolean')
                                else:
                                    ss = s.astype('string').str.strip().str.lower()
                                    mapping = {'true': True, 'false': False, '1': True, '0': False, 'yes': True, 'no': False, 'y': True, 'n': False, 't': True, 'f': False}
                                    df[col] = ss.map(mapping).astype('boolean')
                            except Exception:
                                continue
                        elif dtype == 'category':
                            df[col] = df[col].astype('category')
                        elif dtype == 'string':
                            df[col] = df[col].astype('string')
                    except Exception:
                        # best-effort; leave column unchanged on failure
                        continue
            elif op.get('op') == 'scale':
                cols = op.get('cols', [])
                method = op.get('method')
                params = op.get('params', {}) or {}
                for col in cols:
                    if col not in df.columns:
                        continue
                    try:
                        ser = pd.to_numeric(df[col], errors='coerce').astype(float)
                        if method == 'standard':
                            mean = params.get('mean')
                            std = params.get('std')
                            if mean is None or std is None:
                                mean = float(ser.mean())
                                std = float(ser.std()) if float(ser.std()) != 0 else 1.0
                            df[col] = (ser - mean) / (std if std != 0 else 1.0)
                        elif method == 'minmax':
                            mn = params.get('min')
                            mx = params.get('max')
                            if mn is None or mx is None:
                                mn = float(ser.min()) if not ser.empty else 0.0
                                mx = float(ser.max()) if not ser.empty else 1.0
                            denom = (mx - mn) if (mx - mn) != 0 else 1.0
                            df[col] = (ser - mn) / denom
                    except Exception:
                        continue
            elif op.get('op') == 'drop_duplicates':
                subset = op.get('subset')
                # stored as [] when no subset => treat as None
                if subset == []:
                    subset = None
                keep = op.get('keep', 'first')
                try:
                    if keep is False:
                        # remove all rows that are duplicated (keep=False)
                        dup_mask = df.duplicated(subset=subset, keep=False)
                        df = df.loc[~dup_mask]
                    else:
                        df = df.drop_duplicates(subset=subset, keep=keep)
                except Exception:
                    continue
            elif op.get('op') == 'set_value':
                idx = op.get('index')
                col = op.get('col')
                val = op.get('val')
                try:
                    if idx in df.index and col in df.columns:
                        # attempt to cast val to column dtype
                        ser = df[col]
                        if pd.api.types.is_numeric_dtype(ser):
                            df.at[idx, col] = pd.to_numeric(val, errors='coerce')
                        elif pd.api.types.is_datetime64_any_dtype(ser):
                            df.at[idx, col] = pd.to_datetime(val, errors='coerce')
                        elif pd.api.types.is_bool_dtype(ser) or str(ser.dtype).startswith('boolean'):
                            v = str(val).strip().lower()
                            if v in ('true','1','yes','y','t'):
                                df.at[idx, col] = True
                            elif v in ('false','0','no','n','f'):
                                df.at[idx, col] = False
                            else:
                                df.at[idx, col] = pd.NA
                        else:
                            df.at[idx, col] = val
                except Exception:
                    continue
        return df

    # Diagnostic tab: KPIs only
    with tab_diag:
        left, right = st.columns(2)
        with left:
            st.header(name_a)
            render_kpis(name_a, info_a, df=df_a)
        with right:
            st.header(name_b)
            render_kpis(name_b, info_b, df=df_b)

        st.markdown('---')

    # Exploration tab: interactive exploration and visualisations (moved out of Diagnostic)
    with tab_explore:
        st.subheader('Exploration interactive')
        dataset = st.selectbox('Choisir un jeu de données pour exploration', [a_path.name, b_path.name])
        # Par défaut, afficher la version nettoyée si elle existe en session,
        # sinon revenir à la version originale. L'utilisateur peut forcer
        # l'affichage de la version originale via la checkbox.
        use_clean = st.checkbox('Afficher la version nettoyée (si disponible)', value=True)
        if use_clean:
            df = st.session_state.get('df_a_clean') if dataset == a_path.name else st.session_state.get('df_b_clean')
            # fallback to original if session-stored cleaned version is missing
            if df is None:
                df = df_a if dataset == a_path.name else df_b
        else:
            df = df_a if dataset == a_path.name else df_b

        # Afficher un petit badge indiquant si la vue utilise la version nettoyée
        if dataset == a_path.name:
            is_clean_view = bool(st.session_state.get('df_a_clean') is not None and use_clean)
        else:
            is_clean_view = bool(st.session_state.get('df_b_clean') is not None and use_clean)
        badge_text = 'Affichage: <b>version nettoyée</b>' if is_clean_view else 'Affichage: <b>version originale</b>'
        badge_color = '#16a34a' if is_clean_view else '#6b7280'
        st.markdown(f"<div style='padding:6px;border-radius:6px;background:{badge_color};color:white;display:inline-block'>{badge_text}</div>", unsafe_allow_html=True)

        # Sélecteur de variable cible (override de la détection automatique)
        key_target = 'selected_target_a' if dataset == a_path.name else 'selected_target_b'
        detected_tgt = detect_target(df)
        tgt_options = ['(aucune)'] + list(df.columns)
        default_index = 0
        if detected_tgt in tgt_options:
            default_index = tgt_options.index(detected_tgt)
        # initialize session state key if absent
        if key_target not in st.session_state:
            st.session_state[key_target] = detected_tgt if detected_tgt in df.columns else '(aucune)'
        selected_tgt = st.selectbox('Variable cible (override automatique)', tgt_options, index=default_index, key=key_target)

        # Section A: Variables numériques — distribution et relation avec la cible
        st.subheader('Variables numériques — distribution et relation avec la cible')
        num_cols = df.select_dtypes(include='number').columns.tolist()
        if not num_cols:
            st.info('Aucune colonne numérique détectée dans ce jeu.')
        else:
            num_choice = st.selectbox('Choisir une colonne numérique à visualiser', [''] + num_cols, key='diag_num_choice')
            if num_choice:
                color_seq = ['#16a34a'] if palette == 'Vert / Rouge' else ['#2563eb']
                # histogramme + densité sommaire
                fig = px.histogram(df, x=num_choice, nbins=50, title=f'Distribution: {num_choice}', color_discrete_sequence=color_seq)
                st.plotly_chart(fig, use_container_width=True)

                # boxplot global pour la variable numérique
                try:
                    fig_box_global = px.box(df, y=num_choice, title=f'Boxplot (globale) — {num_choice}', color_discrete_sequence=color_seq)
                    st.plotly_chart(fig_box_global, use_container_width=True)
                except Exception:
                    pass

                # show basic numeric summary
                desc = df[num_choice].describe().to_frame().T
                st.dataframe(desc, use_container_width=True)

                # relation avec la cible si cible numérique sélectionnée
                tgt = st.session_state.get(key_target, '(aucune)')
                if tgt == '(aucune)':
                    tgt = None
                if tgt and tgt != num_choice:
                    if tgt in df.columns and pd.api.types.is_numeric_dtype(df[tgt]):
                        # scatter with regression (numeric vs target)
                        try:
                            import statsmodels  # type: ignore
                            has_sm = True
                        except Exception:
                            has_sm = False
                        if has_sm:
                            fig2 = px.scatter(df, x=num_choice, y=tgt, title=f'{num_choice} vs {tgt}', trendline='ols', color_discrete_sequence=color_seq)
                            st.plotly_chart(fig2, use_container_width=True)
                        else:
                            fig2 = px.scatter(df, x=num_choice, y=tgt, title=f'{num_choice} vs {tgt}', color_discrete_sequence=color_seq)
                            st.plotly_chart(fig2, use_container_width=True)
                            st.info('Affichage sans ligne de tendance — installez `statsmodels` pour afficher la régression (pip install statsmodels).')

                        # Additional: boxplot of target grouped by quantile-bins of the numeric variable
                        try:
                            valid = df[[num_choice, tgt]].dropna()
                            if len(valid) > 0:
                                # use up to 10 quantile bins, drop duplicates when values are constant
                                n_bins = min(10, len(valid))
                                valid = valid.copy()
                                try:
                                    valid['bin'] = pd.qcut(valid[num_choice], q=n_bins, duplicates='drop')
                                except Exception:
                                    # fallback to equal-width bins
                                    valid['bin'] = pd.cut(valid[num_choice], bins=n_bins)
                                fig_box_bins = px.box(valid, x='bin', y=tgt, title=f'Distribution de {tgt} par bins de {num_choice}', labels={'bin': f'Bins de {num_choice}', tgt: tgt})
                                st.plotly_chart(fig_box_bins, use_container_width=True)
                                # show stats per bin
                                stats = valid.groupby('bin')[tgt].agg(['count', 'mean', 'median', 'std']).reset_index()
                                st.dataframe(stats, use_container_width=True)
                        except Exception:
                            pass
                    else:
                        st.info('La variable cible sélectionnée n\'est pas numérique ou n\'existe pas dans ce jeu.')

        # Section B: Variables catégorielles — distribution de la cible par modalité
        st.subheader('Variables catégorielles — comparaison de la cible par modalité')
        cat_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
        if not cat_cols:
            st.info('Aucune colonne catégorielle détectée dans ce jeu.')
        else:
            cat_choice = st.selectbox('Choisir une colonne catégorielle', [''] + cat_cols, key='diag_cat_choice')
            # determine target from session override if present
            tgt_sel = st.session_state.get(key_target, None)
            if tgt_sel == '(aucune)':
                tgt_sel = None
            if cat_choice:
                if not tgt_sel or tgt_sel not in df.columns or not pd.api.types.is_numeric_dtype(df[tgt_sel]):
                    st.info('Sélectionnez d\'abord une variable cible numérique via le sélecteur de cible ci‑dessus.')
                else:
                    # limit categories to top N by count to keep plots readable
                    top_n = 20
                    counts = df[cat_choice].value_counts(dropna=False)
                    top_categories = counts.head(top_n).index.tolist()
                    df_plot = df[df[cat_choice].isin(top_categories)].copy()
                    agg = df_plot.groupby(cat_choice)[tgt_sel].agg(['mean', 'median', 'count']).reset_index()
                    agg = agg.sort_values('count', ascending=False)
                    fig_bar = px.bar(agg, x=cat_choice, y='mean', color='count', title=f'Moyenne de {tgt_sel} par {cat_choice}', labels={'mean': f'Moyenne {tgt_sel}'})
                    st.plotly_chart(fig_bar, use_container_width=True)
                    # boxplot (distribution) if categories not too many
                    try:
                        fig_box = px.box(df_plot, x=cat_choice, y=tgt_sel, title=f'Distribution de {tgt_sel} par {cat_choice}')
                        st.plotly_chart(fig_box, use_container_width=True)
                    except Exception:
                        st.info('Impossible d\'afficher le boxplot pour cette colonne (trop de catégories ou types incompatibles).')

                    # Visualisation: delta de la moyenne par modalité vs moyenne globale (positive / negative)
                    try:
                        grp = df_plot.groupby(cat_choice)[tgt_sel].agg(['count', 'mean', 'median', 'std']).reset_index()
                        global_mean = df[tgt_sel].mean()
                        grp['delta'] = grp['mean'] - global_mean
                        # standard error (NaN possible if count<=1)
                        grp['se'] = grp['std'] / (grp['count'] ** 0.5)
                        grp = grp.sort_values('delta', ascending=False).reset_index(drop=True)
                        grp['sign'] = grp['delta'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))
                        color_map = {'positive': '#16a34a', 'negative': '#dc2626', 'neutral': '#6b7280'}
                        # bar chart of delta (mean - global_mean) with error bars (se)
                        fig_delta = px.bar(grp, x=cat_choice, y='delta', color='sign', color_discrete_map=color_map,
                                           error_y='se', hover_data=['count', 'mean', 'median', 'std'],
                                           title=f'Différence de la moyenne de {tgt_sel} vs moyenne globale — {cat_choice}')
                        fig_delta.update_layout(xaxis={'categoryorder': 'total descending'})
                        st.plotly_chart(fig_delta, use_container_width=True)
                        # display table with key stats for inspection
                        st.dataframe(grp[[cat_choice, 'count', 'mean', 'median', 'std', 'delta']].rename(columns={'delta': 'mean - global_mean'}), use_container_width=True)
                    except Exception:
                        pass

        st.subheader('Vue d\'ensemble — valeurs manquantes')
        miss = df.isna().sum().sort_values(ascending=False).head(30)
        miss_pct = (miss / len(df) * 100).round(2)
        miss_df = pd.DataFrame({'col': miss_pct.index, 'missing_%': miss_pct.values})

        def miss_color(pct):
            if pct < 10:
                return '#16a34a'
            if pct < 50:
                return '#f59e0b'
            return '#dc2626'

        miss_df['color'] = miss_df['missing_%'].map(miss_color)
        fig_m = px.bar(miss_df, x='col', y='missing_%', title='Top 30 colonnes par % de valeurs manquantes', color='color', color_discrete_map='identity')
        fig_m.update_layout(xaxis={'categoryorder':'total descending'}, showlegend=False)
        st.plotly_chart(fig_m, use_container_width=True)

        st.markdown('---')

    # Data Loading tab: allow user to upload or select CSVs for dataset A and B
    with tab_load:
        st.header('Chargement des jeux de données')
        st.markdown('Choisissez ou téléversez deux fichiers CSV à comparer. Si rien n\'est chargé, les jeux par défaut seront utilisés.')
        files = sorted([p.name for p in DATA_DIR.glob('*.csv')])
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader('Jeu A')
            sel_a = st.selectbox('Choisir un CSV existant (data/)', [''] + files, key='sel_a_file')
            up_a = st.file_uploader('Ou téléverser un CSV pour A', type=['csv'], key='upload_a')
            if sel_a:
                path_a = DATA_DIR / sel_a
                try:
                    df_new = pd.read_csv(path_a)
                    st.session_state['df_a_custom'] = df_new
                    st.session_state['name_a'] = sel_a
                    st.session_state['df_a_clean'] = df_new.copy()
                    st.session_state['pipeline_a'] = []
                    st.success(f'Fichier {sel_a} chargé pour Jeu A')
                except Exception as e:
                    st.error(f'Echec lecture {sel_a}: {e}')
            if up_a is not None:
                try:
                    df_new = pd.read_csv(up_a)
                    st.session_state['df_a_custom'] = df_new
                    st.session_state['name_a'] = getattr(up_a, 'name', 'upload_a.csv')
                    st.session_state['df_a_clean'] = df_new.copy()
                    st.session_state['pipeline_a'] = []
                    st.success('CSV uploadé et chargé pour Jeu A')
                except Exception as e:
                    st.error(f'Echec lecture du CSV uploadé: {e}')
        with col_b:
            st.subheader('Jeu B')
            sel_b = st.selectbox('Choisir un CSV existant (data/)', [''] + files, key='sel_b_file')
            up_b = st.file_uploader('Ou téléverser un CSV pour B', type=['csv'], key='upload_b')
            if sel_b:
                path_b2 = DATA_DIR / sel_b
                try:
                    df_new = pd.read_csv(path_b2)
                    st.session_state['df_b_custom'] = df_new
                    st.session_state['name_b'] = sel_b
                    st.session_state['df_b_clean'] = df_new.copy()
                    st.session_state['pipeline_b'] = []
                    st.success(f'Fichier {sel_b} chargé pour Jeu B')
                except Exception as e:
                    st.error(f'Echec lecture {sel_b}: {e}')
            if up_b is not None:
                try:
                    df_new = pd.read_csv(up_b)
                    st.session_state['df_b_custom'] = df_new
                    st.session_state['name_b'] = getattr(up_b, 'name', 'upload_b.csv')
                    st.session_state['df_b_clean'] = df_new.copy()
                    st.session_state['pipeline_b'] = []
                    st.success('CSV uploadé et chargé pour Jeu B')
                except Exception as e:
                    st.error(f'Echec lecture du CSV uploadé: {e}')

        st.markdown('---')

    # compute simple scores (used in Comparison tab)
    score_a = 0.0
    score_b = 0.0
    tgt_a = detect_target(df_a) is not None
    tgt_b = detect_target(df_b) is not None
    if tgt_a:
        score_a += w_target
    if tgt_b:
        score_b += w_target
    rows_max = max(info_a['rows'], info_b['rows'])
    score_a += w_rows * (info_a['rows'] / rows_max)
    score_b += w_rows * (info_b['rows'] / rows_max)
    # Comparison tab will render the recommendation and visual comparison (below)
    with tab_clean:
        st.header('Nettoyage — actions interactives')
        st.markdown('Prévisualisez et appliquez des opérations simples de nettoyage. Les modifications sont conservées en session ; utilisez *Exporter* pour sauvegarder un CSV.')
        # Controls: select dataset to clean and import options
        ds_clean = st.selectbox('Choisir jeu à nettoyer', [a_path.name, b_path.name])
        df_clean = st.session_state['df_a_clean'] if ds_clean == a_path.name else st.session_state['df_b_clean']

        # Badge indiquant si la version affichée est la version nettoyée (pipeline présent)
        pipeline = st.session_state.get('pipeline_a') if ds_clean == a_path.name else st.session_state.get('pipeline_b')
        is_clean_view = bool(pipeline and len(pipeline) > 0)
        if is_clean_view:
            ptxt = f"Affichage: <b>version nettoyée</b> — opérations en session: {len(pipeline)}"
            pcol = '#16a34a'
        else:
            ptxt = 'Affichage: <b>version originale / nettoyée (sans opérations)</b>'
            pcol = '#6b7280'
        st.markdown(f"<div style='padding:6px;border-radius:6px;background:{pcol};color:white;display:inline-block'>{ptxt}</div>", unsafe_allow_html=True)

        # Import cleaned CSV: either pick an existing file in `data/` or upload one
        st.markdown('**Importer / Charger une version nettoyée**')
        try:
            cleaned_files = sorted([p.name for p in DATA_DIR.glob('*.csv') if 'clean' in p.name.lower()])
        except Exception:
            cleaned_files = []
        if cleaned_files:
            sel_file = st.selectbox('Fichiers nettoyés trouvés dans `data/`', [''] + cleaned_files, key='sel_clean_file')
            if sel_file:
                if st.button('Charger ce fichier en session', key='load_existing'):
                    path = DATA_DIR / sel_file
                    try:
                        df_new = pd.read_csv(path)
                        if ds_clean == a_path.name:
                            st.session_state['df_a_clean'] = df_new
                            st.session_state['pipeline_a'] = []
                        else:
                            st.session_state['df_b_clean'] = df_new
                            st.session_state['pipeline_b'] = []
                        st.success(f"{sel_file} chargé en session pour {ds_clean}")
                        # refresh df_clean reference
                        df_clean = st.session_state['df_a_clean'] if ds_clean == a_path.name else st.session_state['df_b_clean']
                    except Exception as e:
                        st.error(f"Échec lecture CSV: {e}")
        uploaded = st.file_uploader('Ou téléverser un CSV nettoyé (remplace la version en session)', type=['csv'], key='upload_clean')
        if uploaded is not None:
            try:
                df_new = pd.read_csv(uploaded)
                if ds_clean == a_path.name:
                    st.session_state['df_a_clean'] = df_new
                    st.session_state['pipeline_a'] = []
                else:
                    st.session_state['df_b_clean'] = df_new
                    st.session_state['pipeline_b'] = []
                st.success('CSV chargé en session depuis l\'upload.')
                df_clean = st.session_state['df_a_clean'] if ds_clean == a_path.name else st.session_state['df_b_clean']
            except Exception as e:
                st.error(f"Échec lecture du CSV uploadé: {e}")

        st.subheader('Aperçu (preview)')
        # Sélecteur de taille d'aperçu pour éviter de rendre tout le DF (performance)
        preview_choice = st.selectbox('Taille de l\'aperçu',
                                      ['50', '200', '500', '1000', 'Échantillon aléatoire (200)', 'Tous'],
                                      index=1,
                                      key=f'preview_size_{ds_clean}')

        def _make_preview(df: pd.DataFrame, choice: str) -> pd.DataFrame:
            try:
                if choice == 'Tous':
                    return df.copy()
                if choice.startswith('Échantillon'):
                    n = 200
                    return df.sample(n=min(n, len(df)), random_state=42)
                n = int(choice)
                return df.head(n)
            except Exception:
                return df.head(200)

        # Afficher le pourcentage de complétude (non manquants) pour chaque colonne
        try:
            def _dtype_short(col_name: str) -> str:
                ser = df_clean[col_name]
                try:
                    if pd.api.types.is_integer_dtype(ser):
                        return 'int'
                    if pd.api.types.is_float_dtype(ser):
                        return 'float'
                    if pd.api.types.is_datetime64_any_dtype(ser):
                        return 'datetime'
                    if pd.api.types.is_bool_dtype(ser) or str(ser.dtype).startswith('boolean'):
                        return 'bool'
                    if pd.api.types.is_categorical_dtype(ser):
                        return 'category'
                except Exception:
                    pass
                return 'string'

            col_display = {col: f"{col}\n({_dtype_short(col)}) — compl.: {df_clean[col].notna().mean()*100:.1f}%" for col in df_clean.columns}
            preview_df = _make_preview(df_clean, preview_choice).rename(columns=col_display)
            st.dataframe(preview_df, use_container_width=True)
        except Exception:
            # Fallback simple view
            st.dataframe(_make_preview(df_clean, preview_choice), use_container_width=True)

        st.subheader('Opérations rapides')
        c1, c2 = st.columns(2)
        with c1:
            drop_cols = st.multiselect('Supprimer colonnes (sélection)', list(df_clean.columns))
            if st.button('Prévisualiser suppression'):
                if not drop_cols:
                    st.info('Aucune colonne sélectionnée pour suppression.')
                else:
                    preview = _make_preview(df_clean, preview_choice).copy()
                    # highlight selected columns in light red
                    def _highlight(col: pd.Series):
                        if col.name in drop_cols:
                            return ['background-color: #ffdddd' for _ in col]
                        return ['' for _ in col]
                    styled = preview.style.apply(_highlight, axis=0)
                    # render styled HTML using components to preserve styling
                    components.html(styled.to_html(), height=400, scrolling=True)
            if st.button('Appliquer suppression'):
                if drop_cols:
                    if ds_clean == a_path.name:
                        st.session_state['df_a_clean'] = df_clean.drop(columns=drop_cols, errors='ignore')
                        st.session_state['pipeline_a'].append({'op':'drop', 'cols': drop_cols})
                    else:
                        st.session_state['df_b_clean'] = df_clean.drop(columns=drop_cols, errors='ignore')
                        st.session_state['pipeline_b'].append({'op':'drop', 'cols': drop_cols})
                    st.success('Suppression appliquée en session.')
                else:
                    st.info('Aucune colonne sélectionnée — rien à appliquer.')
            # Suppression de lignes
            st.markdown('**Supprimer des lignes**')
            row_mode = st.radio('Mode', ['Par index (sélection)', 'Par condition (pandas query)'], key=f'row_del_mode_{ds_clean}')
            if row_mode.startswith('Par index'):
                preview_idx = _make_preview(df_clean, preview_choice).index.tolist()
                sel_rows = st.multiselect('Sélectionner des index à supprimer (aperçu)', preview_idx, key=f'sel_rows_{ds_clean}')
                if st.button('Prévisualiser suppression de lignes'):
                    if not sel_rows:
                        st.info('Aucun index sélectionné pour suppression.')
                    else:
                        preview = _make_preview(df_clean, preview_choice).copy()
                        mask = preview.index.isin(sel_rows)
                        def _hl_row(row):
                            return ['background-color: #ffdddd' if mask.loc[row.name] else '' for _ in row]
                        components.html(preview.style.apply(_hl_row, axis=1).to_html(), height=400, scrolling=True)
                if st.button('Appliquer suppression de lignes'):
                    if not sel_rows:
                        st.info('Aucun index sélectionné — rien à appliquer.')
                    else:
                        if ds_clean == a_path.name:
                            st.session_state['df_a_clean'] = df_clean.drop(index=sel_rows, errors='ignore')
                            st.session_state['pipeline_a'].append({'op': 'drop_rows', 'index': sel_rows})
                        else:
                            st.session_state['df_b_clean'] = df_clean.drop(index=sel_rows, errors='ignore')
                            st.session_state['pipeline_b'].append({'op': 'drop_rows', 'index': sel_rows})
                        st.success(f'{len(sel_rows)} lignes supprimées et ajoutées au pipeline.')
            else:
                cond = st.text_input('Condition pandas (ex: price > 100000 and operation == "sale")', key=f'row_cond_{ds_clean}')
                if st.button('Prévisualiser suppression par condition'):
                    if not cond:
                        st.info('Aucune condition fournie.')
                    else:
                        try:
                            matched = df_clean.query(cond)
                            if matched.empty:
                                st.info('Aucune ligne ne correspond à la condition.')
                            else:
                                def _hl_cond(row):
                                    try:
                                        return ['background-color: #ffdddd' if row.name in matched.index else '' for _ in row]
                                    except Exception:
                                        return ['' for _ in row]
                                components.html(_make_preview(df_clean, preview_choice).style.apply(_hl_cond, axis=1).to_html(), height=400, scrolling=True)
                        except Exception as e:
                            st.error(f'Condition invalide: {e}')
                if st.button('Appliquer suppression par condition'):
                    if not cond:
                        st.info('Aucune condition fournie — rien à appliquer.')
                    else:
                        try:
                            matched_idx = df_clean.query(cond).index
                            if ds_clean == a_path.name:
                                st.session_state['df_a_clean'] = df_clean.drop(index=matched_idx, errors='ignore')
                                st.session_state['pipeline_a'].append({'op': 'drop_rows_cond', 'condition': cond})
                            else:
                                st.session_state['df_b_clean'] = df_clean.drop(index=matched_idx, errors='ignore')
                                st.session_state['pipeline_b'].append({'op': 'drop_rows_cond', 'condition': cond})
                            st.success(f'{len(matched_idx)} lignes supprimées et ajoutées au pipeline.')
                        except Exception as e:
                            st.error(f'Erreur application condition: {e}')
            # Options pour supprimer automatiquement colonnes complètement vides (0%) ou entièrement complètes (100%)
            st.markdown('**Suppression automatique selon complétude**')
            comp = df_clean.notna().mean()
            empty_cols = comp[comp == 0.0].index.tolist()
            full_cols = comp[comp == 1.0].index.tolist()
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button('Prévisualiser colonnes vides (compl.: 0%)'):
                    if not empty_cols:
                        st.info('Aucune colonne complètement vide.')
                    else:
                        preview = _make_preview(df_clean, preview_choice).copy()
                        def _hl_empty(col: pd.Series):
                            if col.name in empty_cols:
                                return ['background-color: #ffdddd' for _ in col]
                            return ['' for _ in col]
                        components.html(preview.style.apply(_hl_empty, axis=0).to_html(), height=400, scrolling=True)
                if st.button('Supprimer colonnes vides (0%)'):
                    if not empty_cols:
                        st.info('Aucune colonne à supprimer.')
                    else:
                        if ds_clean == a_path.name:
                            st.session_state['df_a_clean'] = df_clean.drop(columns=empty_cols, errors='ignore')
                            st.session_state['pipeline_a'].append({'op':'drop', 'cols': empty_cols})
                        else:
                            st.session_state['df_b_clean'] = df_clean.drop(columns=empty_cols, errors='ignore')
                            st.session_state['pipeline_b'].append({'op':'drop', 'cols': empty_cols})
                        st.success(f"{len(empty_cols)} colonnes vides supprimées et ajoutées au pipeline.")
            with col_b:
                if st.button('Prévisualiser colonnes complètes (compl.: 100%)'):
                    if not full_cols:
                        st.info('Aucune colonne entièrement complète (100%).')
                    else:
                        preview = _make_preview(df_clean, preview_choice).copy()
                        def _hl_full(col: pd.Series):
                            if col.name in full_cols:
                                return ['background-color: #ffdddd' for _ in col]
                            return ['' for _ in col]
                        components.html(preview.style.apply(_hl_full, axis=0).to_html(), height=400, scrolling=True)
                if st.button('Supprimer colonnes complètes (100%)'):
                    if not full_cols:
                        st.info('Aucune colonne à supprimer.')
                    else:
                        if ds_clean == a_path.name:
                            st.session_state['df_a_clean'] = df_clean.drop(columns=full_cols, errors='ignore')
                            st.session_state['pipeline_a'].append({'op':'drop', 'cols': full_cols})
                        else:
                            st.session_state['df_b_clean'] = df_clean.drop(columns=full_cols, errors='ignore')
                            st.session_state['pipeline_b'].append({'op':'drop', 'cols': full_cols})
                        st.success(f"{len(full_cols)} colonnes complètes supprimées et ajoutées au pipeline.")
            # Supprimer toutes les lignes où une colonne sélectionnée est manquante
            st.markdown('**Supprimer lignes avec valeurs manquantes (colonne)**')
            na_col = st.selectbox('Choisir une colonne (supprimer lignes où NA)', [''] + list(df_clean.columns), key=f'na_col_{ds_clean}')
            if na_col:
                if st.button('Prévisualiser lignes manquantes pour cette colonne'):
                    preview = _make_preview(df_clean, preview_choice).copy()
                    mask = preview[na_col].isna()
                    if mask.any():
                        def _hl_na(row):
                            return ['background-color: #ffdddd' if row.name in preview[mask].index else '' for _ in row]
                        components.html(preview.style.apply(_hl_na, axis=1).to_html(), height=400, scrolling=True)
                    else:
                        st.info('Aucune valeur manquante trouvée dans l\'aperçu pour cette colonne.')
                if st.button('Supprimer toutes les lignes où cette colonne est manquante'):
                    try:
                        matched_idx = df_clean[df_clean[na_col].isna()].index
                        if matched_idx.empty:
                            st.info('Aucune ligne à supprimer (aucune valeur manquante).')
                        else:
                            if ds_clean == a_path.name:
                                st.session_state['df_a_clean'] = df_clean.drop(index=matched_idx, errors='ignore')
                                st.session_state['pipeline_a'].append({'op': 'drop_rows_where_na', 'col': na_col})
                            else:
                                st.session_state['df_b_clean'] = df_clean.drop(index=matched_idx, errors='ignore')
                                st.session_state['pipeline_b'].append({'op': 'drop_rows_where_na', 'col': na_col})
                            st.success(f'{len(matched_idx)} lignes supprimées et ajoutées au pipeline.')
                    except Exception as e:
                        st.error(f'Erreur lors de la suppression: {e}')
        with c2:
            num_cols = df_clean.select_dtypes(include='number').columns.tolist()
            impute_col = st.selectbox('Imputer (numérique) — choisir colonne', [''] + num_cols)
            impute_strategy = st.selectbox('Stratégie', ['médiane', 'moyenne', 'constante'])
            const_val = None
            if impute_strategy == 'constante':
                const_val = st.text_input('Valeur constante (ex: 0)')
            if st.button('Prévisualiser imputation') and impute_col:
                if impute_strategy == 'médiane':
                    val = df_clean[impute_col].median()
                elif impute_strategy == 'moyenne':
                    val = df_clean[impute_col].mean()
                else:
                    try:
                        val = float(const_val)
                    except Exception:
                        val = const_val
                preview = _make_preview(df_clean, preview_choice).copy()
                preview[impute_col] = preview[impute_col].fillna(val)
                try:
                    def _dtype_short_local(col_name: str) -> str:
                        ser = df_clean[col_name]
                        try:
                            if pd.api.types.is_integer_dtype(ser):
                                return 'int'
                            if pd.api.types.is_float_dtype(ser):
                                return 'float'
                            if pd.api.types.is_datetime64_any_dtype(ser):
                                return 'datetime'
                            if pd.api.types.is_bool_dtype(ser) or str(ser.dtype).startswith('boolean'):
                                return 'bool'
                            if pd.api.types.is_categorical_dtype(ser):
                                return 'category'
                        except Exception:
                            pass
                        return 'string'

                    col_display = {impute_col: f"{impute_col}\n({_dtype_short_local(impute_col)}) — compl.: {df_clean[impute_col].notna().mean()*100:.1f}%"}
                    st.dataframe(preview[[impute_col]].rename(columns=col_display), use_container_width=True)
                except Exception:
                    st.dataframe(preview[[impute_col]], use_container_width=True)
            if st.button('Appliquer imputation') and impute_col:
                if impute_strategy == 'médiane':
                    val = df_clean[impute_col].median()
                elif impute_strategy == 'moyenne':
                    val = df_clean[impute_col].mean()
                else:
                    try:
                        val = float(const_val)
                    except Exception:
                        val = const_val
                if ds_clean == a_path.name:
                    st.session_state['df_a_clean'][impute_col] = st.session_state['df_a_clean'][impute_col].fillna(val)
                    st.session_state['pipeline_a'].append({'op':'impute', 'col': impute_col, 'val': val})
                else:
                    st.session_state['df_b_clean'][impute_col] = st.session_state['df_b_clean'][impute_col].fillna(val)
                    st.session_state['pipeline_b'].append({'op':'impute', 'col': impute_col, 'val': val})
                st.success('Imputation appliquée en session.')

            # Typage & Normalisation
            st.markdown('**Typage & Normalisation**')
            # Typage (caster les colonnes)
            cast_cols = st.multiselect('Colonnes à caster', list(df_clean.columns), key=f'cast_cols_{ds_clean}')
            cast_dtype = st.selectbox('Type cible', ['int', 'float', 'datetime', 'bool', 'category', 'string'], key=f'cast_dtype_{ds_clean}')
            if st.button('Prévisualiser typage') and cast_cols:
                preview = _make_preview(df_clean, preview_choice).copy()
                for col in cast_cols:
                    try:
                        if cast_dtype == 'int':
                            preview[col] = pd.to_numeric(preview[col], errors='coerce').astype('Int64')
                        elif cast_dtype == 'float':
                            preview[col] = pd.to_numeric(preview[col], errors='coerce').astype(float)
                        elif cast_dtype == 'datetime':
                            preview[col] = pd.to_datetime(preview[col], errors='coerce')
                        elif cast_dtype == 'category':
                            preview[col] = preview[col].astype('category')
                        elif cast_dtype == 'string':
                            preview[col] = preview[col].astype('string')
                        elif cast_dtype == 'bool':
                            try:
                                s = preview[col]
                                # numeric -> bool, else try common string mappings
                                if pd.api.types.is_numeric_dtype(s):
                                    preview[col] = s.fillna(0).astype(int).astype(bool).astype('boolean')
                                else:
                                    ss = s.astype('string').str.strip().str.lower()
                                    mapping = {'true': True, 'false': False, '1': True, '0': False, 'yes': True, 'no': False, 'y': True, 'n': False, 't': True, 'f': False}
                                    preview[col] = ss.map(mapping).astype('boolean')
                            except Exception:
                                continue
                    except Exception:
                        continue
                try:
                    def _dtype_short_preview(col_name: str) -> str:
                        ser = df_clean[col_name] if col_name in df_clean.columns else preview[col_name]
                        try:
                            if pd.api.types.is_integer_dtype(ser):
                                return 'int'
                            if pd.api.types.is_float_dtype(ser):
                                return 'float'
                            if pd.api.types.is_datetime64_any_dtype(ser):
                                return 'datetime'
                            if pd.api.types.is_bool_dtype(ser) or str(ser.dtype).startswith('boolean'):
                                return 'bool'
                            if pd.api.types.is_categorical_dtype(ser):
                                return 'category'
                        except Exception:
                            pass
                        return 'string'

                    col_display = {col: f"{col}\n({_dtype_short_preview(col)}) — compl.: {df_clean[col].notna().mean()*100:.1f}%" for col in preview.columns}
                    st.dataframe(preview.rename(columns=col_display), use_container_width=True)
                except Exception:
                    st.dataframe(preview, use_container_width=True)
            if st.button('Appliquer typage') and cast_cols:
                for col in cast_cols:
                    try:
                        if cast_dtype == 'int':
                            if ds_clean == a_path.name:
                                st.session_state['df_a_clean'][col] = pd.to_numeric(st.session_state['df_a_clean'][col], errors='coerce').astype('Int64')
                            else:
                                st.session_state['df_b_clean'][col] = pd.to_numeric(st.session_state['df_b_clean'][col], errors='coerce').astype('Int64')
                        elif cast_dtype == 'float':
                            if ds_clean == a_path.name:
                                st.session_state['df_a_clean'][col] = pd.to_numeric(st.session_state['df_a_clean'][col], errors='coerce').astype(float)
                            else:
                                st.session_state['df_b_clean'][col] = pd.to_numeric(st.session_state['df_b_clean'][col], errors='coerce').astype(float)
                        elif cast_dtype == 'datetime':
                            if ds_clean == a_path.name:
                                st.session_state['df_a_clean'][col] = pd.to_datetime(st.session_state['df_a_clean'][col], errors='coerce')
                            else:
                                st.session_state['df_b_clean'][col] = pd.to_datetime(st.session_state['df_b_clean'][col], errors='coerce')
                        elif cast_dtype == 'category':
                            if ds_clean == a_path.name:
                                st.session_state['df_a_clean'][col] = st.session_state['df_a_clean'][col].astype('category')
                            else:
                                st.session_state['df_b_clean'][col] = st.session_state['df_b_clean'][col].astype('category')
                        elif cast_dtype == 'string':
                            if ds_clean == a_path.name:
                                st.session_state['df_a_clean'][col] = st.session_state['df_a_clean'][col].astype('string')
                            else:
                                st.session_state['df_b_clean'][col] = st.session_state['df_b_clean'][col].astype('string')
                        elif cast_dtype == 'bool':
                            try:
                                if ds_clean == a_path.name:
                                    s = st.session_state['df_a_clean'][col]
                                else:
                                    s = st.session_state['df_b_clean'][col]
                                if pd.api.types.is_numeric_dtype(s):
                                    converted = s.fillna(0).astype(int).astype(bool).astype('boolean')
                                else:
                                    ss = s.astype('string').str.strip().str.lower()
                                    mapping = {'true': True, 'false': False, '1': True, '0': False, 'yes': True, 'no': False, 'y': True, 'n': False, 't': True, 'f': False}
                                    converted = ss.map(mapping).astype('boolean')
                                if ds_clean == a_path.name:
                                    st.session_state['df_a_clean'][col] = converted
                                else:
                                    st.session_state['df_b_clean'][col] = converted
                            except Exception:
                                continue
                    except Exception:
                        continue
                # append single cast op describing columns and dtype
                if ds_clean == a_path.name:
                    st.session_state['pipeline_a'].append({'op': 'cast', 'cols': cast_cols, 'dtype': cast_dtype})
                else:
                    st.session_state['pipeline_b'].append({'op': 'cast', 'cols': cast_cols, 'dtype': cast_dtype})
                st.success('Typage appliqué et ajouté au pipeline.')

            # Normalisation (scaling)
            st.markdown('**Normalisation (colonnes numériques)**')
            scale_cols = st.multiselect('Colonnes numériques à normaliser', num_cols, key=f'scale_cols_{ds_clean}')
            scale_method = st.selectbox('Méthode de normalisation', ['Standard (z-score)', 'Min-Max (0-1)'], key=f'scale_method_{ds_clean}')
            method_key = 'standard' if scale_method.startswith('Standard') else 'minmax'
            if st.button('Prévisualiser normalisation') and scale_cols:
                preview = _make_preview(df_clean, preview_choice).copy()
                for col in scale_cols:
                    try:
                        ser = pd.to_numeric(preview[col], errors='coerce').astype(float)
                        if method_key == 'standard':
                            mean = float(ser.mean()) if not ser.dropna().empty else 0.0
                            std = float(ser.std()) if float(ser.std()) != 0 else 1.0
                            preview[col] = (ser - mean) / (std if std != 0 else 1.0)
                        else:
                            mn = float(ser.min()) if not ser.dropna().empty else 0.0
                            mx = float(ser.max()) if not ser.dropna().empty else 1.0
                            denom = (mx - mn) if (mx - mn) != 0 else 1.0
                            preview[col] = (ser - mn) / denom
                    except Exception:
                        continue
                try:
                    def _dtype_short_preview2(col_name: str) -> str:
                        ser = df_clean[col_name] if col_name in df_clean.columns else preview[col_name]
                        try:
                            if pd.api.types.is_integer_dtype(ser):
                                return 'int'
                            if pd.api.types.is_float_dtype(ser):
                                return 'float'
                            if pd.api.types.is_datetime64_any_dtype(ser):
                                return 'datetime'
                            if pd.api.types.is_bool_dtype(ser) or str(ser.dtype).startswith('boolean'):
                                return 'bool'
                            if pd.api.types.is_categorical_dtype(ser):
                                return 'category'
                        except Exception:
                            pass
                        return 'string'

                    col_display = {col: f"{col}\n({_dtype_short_preview2(col)}) — compl.: {df_clean[col].notna().mean()*100:.1f}%" for col in preview.columns}
                    st.dataframe(preview.rename(columns=col_display), use_container_width=True)
                except Exception:
                    st.dataframe(preview, use_container_width=True)
            if st.button('Appliquer normalisation') and scale_cols:
                # compute params on full df_clean and apply to session df
                params = {}
                for col in scale_cols:
                    try:
                        ser_full = pd.to_numeric((st.session_state['df_a_clean'] if ds_clean == a_path.name else st.session_state['df_b_clean'])[col], errors='coerce').astype(float)
                        if method_key == 'standard':
                            mean = float(ser_full.mean()) if not ser_full.dropna().empty else 0.0
                            std = float(ser_full.std()) if float(ser_full.std()) != 0 else 1.0
                            params[col] = {'mean': mean, 'std': std}
                            transformed = (ser_full - mean) / (std if std != 0 else 1.0)
                        else:
                            mn = float(ser_full.min()) if not ser_full.dropna().empty else 0.0
                            mx = float(ser_full.max()) if not ser_full.dropna().empty else 1.0
                            params[col] = {'min': mn, 'max': mx}
                            denom = (mx - mn) if (mx - mn) != 0 else 1.0
                            transformed = (ser_full - mn) / denom
                        if ds_clean == a_path.name:
                            st.session_state['df_a_clean'][col] = transformed
                        else:
                            st.session_state['df_b_clean'][col] = transformed
                    except Exception:
                        continue
                # append a single scale op with params dict
                if ds_clean == a_path.name:
                    st.session_state['pipeline_a'].append({'op': 'scale', 'cols': scale_cols, 'method': method_key, 'params': params})
                else:
                    st.session_state['pipeline_b'].append({'op': 'scale', 'cols': scale_cols, 'method': method_key, 'params': params})
                st.success('Normalisation appliquée et ajoutée au pipeline.')

        # Détection et suppression des doublons
        st.markdown('**Doublons (lignes dupliquées)**')
        dup_subset = st.multiselect('Colonnes à considérer pour doublons (laisser vide = toutes colonnes)', list(df_clean.columns), key=f'dup_subset_{ds_clean}')
        dup_mode = st.selectbox('Action sur doublons', ['Marquer / Prévisualiser', 'Supprimer (garder premier)', 'Supprimer (garder dernier)', 'Supprimer toutes les lignes dupliquées)'], index=0, key=f'dup_mode_{ds_clean}')
        # compute duplicate mask on full df_clean
        try:
            if dup_subset:
                dup_mask_full = df_clean.duplicated(subset=dup_subset, keep=False)
            else:
                dup_mask_full = df_clean.duplicated(keep=False)
        except Exception:
            dup_mask_full = pd.Series([False] * len(df_clean), index=df_clean.index)

        col_dup_a, col_dup_b = st.columns(2)
        with col_dup_a:
            if st.button('Prévisualiser doublons'):
                # show preview with duplicated rows highlighted
                preview = _make_preview(df_clean, preview_choice).copy()
                mask = dup_mask_full.reindex(preview.index, fill_value=False)
                def _hl_dup(row):
                    if mask.loc[row.name]:
                        return ['background-color: #ffdddd' for _ in row]
                    return ['' for _ in row]
                components.html(preview.style.apply(_hl_dup, axis=1).to_html(), height=400, scrolling=True)
            if st.button('Compter doublons'):
                total_dup_groups = dup_mask_full.sum()
                st.info(f'Lignes impliquées dans des doublons (keep=False): {int(total_dup_groups)} / {len(df_clean)}')
        with col_dup_b:
            if st.button('Supprimer doublons (garder premier)'):
                if dup_subset:
                    cols = dup_subset
                else:
                    cols = None
                if ds_clean == a_path.name:
                    st.session_state['df_a_clean'] = df_clean.drop_duplicates(subset=cols, keep='first')
                    st.session_state['pipeline_a'].append({'op': 'drop_duplicates', 'subset': cols or [], 'keep': 'first'})
                else:
                    st.session_state['df_b_clean'] = df_clean.drop_duplicates(subset=cols, keep='first')
                    st.session_state['pipeline_b'].append({'op': 'drop_duplicates', 'subset': cols or [], 'keep': 'first'})
                st.success('Doublons supprimés (garder premier) et ajoutés au pipeline.')
            if st.button('Supprimer doublons (garder dernier)'):
                if dup_subset:
                    cols = dup_subset
                else:
                    cols = None
                if ds_clean == a_path.name:
                    st.session_state['df_a_clean'] = df_clean.drop_duplicates(subset=cols, keep='last')
                    st.session_state['pipeline_a'].append({'op': 'drop_duplicates', 'subset': cols or [], 'keep': 'last'})
                else:
                    st.session_state['df_b_clean'] = df_clean.drop_duplicates(subset=cols, keep='last')
                    st.session_state['pipeline_b'].append({'op': 'drop_duplicates', 'subset': cols or [], 'keep': 'last'})
                st.success('Doublons supprimés (garder dernier) et ajoutés au pipeline.')
            if st.button('Supprimer toutes lignes dupliquées'):
                if dup_subset:
                    cols = dup_subset
                else:
                    cols = None
                # keep=False removes all rows that are duplicated
                if ds_clean == a_path.name:
                    st.session_state['df_a_clean'] = df_clean.loc[~dup_mask_full].copy()
                    st.session_state['pipeline_a'].append({'op': 'drop_duplicates', 'subset': cols or [], 'keep': False})
                else:
                    st.session_state['df_b_clean'] = df_clean.loc[~dup_mask_full].copy()
                    st.session_state['pipeline_b'].append({'op': 'drop_duplicates', 'subset': cols or [], 'keep': False})
                st.success('Toutes les lignes en doublon supprimées et ajoutées au pipeline.')

        # Édition d'une cellule spécifique
        st.markdown('**Édition — modifier une valeur de cellule**')
        # Provide selection of a row (by index) from the preview and allow manual index entry
        preview_idx = _make_preview(df_clean, preview_choice).index.astype(str).tolist()
        sel_row = st.selectbox('Sélectionner une ligne (index)', [''] + preview_idx, key=f'edit_row_select_{ds_clean}')
        manual_idx = st.text_input('Ou saisir un index (laisser vide si non)', key=f'edit_row_manual_{ds_clean}')
        # determine final index to use
        chosen_idx = None
        if manual_idx:
            try:
                # try to cast to original index types if numeric
                if manual_idx.isdigit():
                    chosen_idx = int(manual_idx)
                else:
                    chosen_idx = manual_idx
            except Exception:
                chosen_idx = manual_idx
        elif sel_row:
            # convert back to original index type where possible
            try:
                # try int
                if sel_row.isdigit():
                    chosen_idx = int(sel_row)
                else:
                    chosen_idx = sel_row
            except Exception:
                chosen_idx = sel_row

        edit_col = st.selectbox('Colonne à modifier', [''] + list(df_clean.columns), key=f'edit_col_{ds_clean}')
        new_val = st.text_input('Nouvelle valeur (saisie texte)', key=f'edit_val_{ds_clean}')
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            if st.button('Prévisualiser modification'):
                if chosen_idx is None or edit_col == '':
                    st.info('Sélectionnez d\'abord une ligne et une colonne.')
                else:
                    if chosen_idx not in df_clean.index:
                        st.error('Index non trouvé dans le jeu de données affiché.')
                    else:
                        before = df_clean.loc[[chosen_idx], [edit_col]].copy()
                        after = before.copy()
                        # attempt best-effort cast based on column dtype
                        def _cast_preview(col_series, val_str):
                            try:
                                if pd.api.types.is_numeric_dtype(col_series):
                                    return pd.to_numeric(val_str, errors='coerce')
                                if pd.api.types.is_datetime64_any_dtype(col_series):
                                    return pd.to_datetime(val_str, errors='coerce')
                                if pd.api.types.is_bool_dtype(col_series) or str(col_series.dtype).startswith('boolean'):
                                    v = str(val_str).strip().lower()
                                    return True if v in ('true','1','yes','y','t') else False if v in ('false','0','no','n','f') else pd.NA
                                return val_str
                            except Exception:
                                return val_str
                        casted = _cast_preview(df_clean[edit_col], new_val)
                        after.at[chosen_idx, edit_col] = casted
                        # show side-by-side
                        combined = pd.concat([before.rename(columns={edit_col: f'{edit_col} (avant)'}), after.rename(columns={edit_col: f'{edit_col} (après)'})], axis=1)
                        def _hl_changed(col: pd.Series):
                            return ['background-color: #ffdddd' if col.name.endswith('(après)') else '' for _ in col]
                        components.html(combined.style.to_html(), height=200, scrolling=True)
        with col_e2:
            if st.button('Appliquer modification'):
                if chosen_idx is None or edit_col == '' or new_val == '':
                    st.info('Sélectionnez une ligne, une colonne et fournissez une nouvelle valeur.')
                else:
                    if chosen_idx not in df_clean.index:
                        st.error('Index non trouvé — impossible d\'appliquer.')
                    else:
                        # apply with best-effort casting
                        def _cast_apply(col_series, val_str):
                            try:
                                if pd.api.types.is_numeric_dtype(col_series):
                                    return pd.to_numeric(val_str, errors='coerce')
                                if pd.api.types.is_datetime64_any_dtype(col_series):
                                    return pd.to_datetime(val_str, errors='coerce')
                                if pd.api.types.is_bool_dtype(col_series) or str(col_series.dtype).startswith('boolean'):
                                    v = str(val_str).strip().lower()
                                    if v in ('true','1','yes','y','t'):
                                        return True
                                    if v in ('false','0','no','n','f'):
                                        return False
                                    return pd.NA
                                return val_str
                            except Exception:
                                return val_str
                        casted_val = _cast_apply(df_clean[edit_col], new_val)
                        if ds_clean == a_path.name:
                            st.session_state['df_a_clean'].at[chosen_idx, edit_col] = casted_val
                            st.session_state['pipeline_a'].append({'op': 'set_value', 'index': chosen_idx, 'col': edit_col, 'val': casted_val})
                        else:
                            st.session_state['df_b_clean'].at[chosen_idx, edit_col] = casted_val
                            st.session_state['pipeline_b'].append({'op': 'set_value', 'index': chosen_idx, 'col': edit_col, 'val': casted_val})
                        st.success('Modification appliquée et ajoutée au pipeline.')

        st.markdown('---')
        # Comparer colonnes — combinaisons
        st.subheader('Comparer colonnes — combinaisons')
        with st.expander('Explorer combinaisons de valeurs', expanded=False):
            comb_primary = st.selectbox('Colonne principale (ex: n_floors)', list(df_clean.columns), key=f'comb_primary_{ds_clean}')
            comb_condition = st.selectbox('Condition', ['Afficher lignes où la colonne principale est manquante', 'Afficher lignes où la colonne principale est non nulle', 'Afficher toutes les lignes'], index=0, key=f'comb_condition_{ds_clean}')
            comb_others = st.multiselect('Colonnes à comparer (au moins une)', [c for c in df_clean.columns if c != comb_primary], key=f'comb_others_{ds_clean}')
            comb_topn = st.number_input('Top N combinaisons à afficher', min_value=5, max_value=200, value=20, step=5, key=f'comb_topn_{ds_clean}')
            if st.button('Analyser combinaisons', key=f'comb_analyze_{ds_clean}'):
                df_local = df_clean.copy()
                if comb_condition.startswith('Afficher lignes où la colonne principale est manquante'):
                    df_local = df_local[df_local[comb_primary].isna()]
                elif comb_condition.startswith('Afficher lignes où la colonne principale est non nulle'):
                    df_local = df_local[df_local[comb_primary].notna()]
                # require at least one other column
                if not comb_others:
                    st.info('Sélectionnez au moins une colonne à comparer.')
                else:
                    # include primary column values as well in the combination
                    cols = [comb_primary] + comb_others
                    combo_series = df_local[cols].fillna('<NA>').astype(str).agg(' | '.join, axis=1)
                    combo_counts = combo_series.value_counts().reset_index()
                    combo_counts.columns = ['combination', 'count']
                    combo_counts['pct'] = combo_counts['count'] / (len(df_local) if len(df_local) > 0 else 1) * 100
                    st.markdown(f"Lignes filtrées: **{len(df_local)}**")
                    st.dataframe(combo_counts.head(comb_topn), use_container_width=True)
                    try:
                        figc = px.bar(combo_counts.head(comb_topn), x='combination', y='count', title='Top combinations', color='count')
                        st.plotly_chart(figc, use_container_width=True)
                    except Exception:
                        pass
                    sel_combo = st.selectbox('Voir exemples pour une combinaison', [''] + combo_counts['combination'].head(comb_topn).tolist(), key=f'comb_select_{ds_clean}')
                    if sel_combo:
                        mask = combo_series == sel_combo
                        st.dataframe(df_local.loc[mask].head(200), use_container_width=True)
                    csv = combo_counts.to_csv(index=False)
                    st.download_button('Télécharger combinaisons (CSV)', csv, file_name=f'combinaisons_{ds_clean}.csv')

        # Legend explaining red highlight and undo actions
        st.markdown('**Légende — Prévisualisation**')
        st.markdown("""
        - **Fond rouge clair** : colonnes sélectionnées pour suppression dans la prévisualisation.
        - **Appliquer suppression** : applique l'opération et l'ajoute au pipeline (session).
        - **Annuler dernière opération** : supprime la dernière étape du pipeline et reconstruit la version nettoyée.
        - **Réinitialiser** : annule tout le pipeline et restaure la version originale.
        """)
        c_undo, c_reset = st.columns(2)
        with c_undo:
            if st.button('Annuler dernière opération'):
                pipeline = st.session_state['pipeline_a'] if ds_clean == a_path.name else st.session_state['pipeline_b']
                if not pipeline:
                    st.info("Aucune opération à annuler.")
                else:
                    pipeline.pop()
                    # rebuild cleaned df from original
                    if ds_clean == a_path.name:
                        st.session_state['df_a_clean'] = apply_pipeline(df_a, pipeline)
                        st.session_state['pipeline_a'] = pipeline
                    else:
                        st.session_state['df_b_clean'] = apply_pipeline(df_b, pipeline)
                        st.session_state['pipeline_b'] = pipeline
                    st.success('Dernière opération annulée.')
        with c_reset:
            if st.button('Réinitialiser tout'):
                if ds_clean == a_path.name:
                    st.session_state['df_a_clean'] = df_a.copy()
                    st.session_state['pipeline_a'] = []
                else:
                    st.session_state['df_b_clean'] = df_b.copy()
                    st.session_state['pipeline_b'] = []
                st.success('Pipeline réinitialisé ; jeu restauré à l\'original.')
        st.subheader('Exporter / Importer')
        if ds_clean == a_path.name:
            csv = st.session_state['df_a_clean'].to_csv(index=False)
            st.download_button('Télécharger version nettoyée (CSV)', csv, file_name=f'{a_path.stem}.cleaned.csv')
        else:
            csv = st.session_state['df_b_clean'].to_csv(index=False)
            st.download_button('Télécharger version nettoyée (CSV)', csv, file_name=f'{b_path.stem}.cleaned.csv')

        # Compte rendu / Rapport de session
        st.markdown('---')
        st.subheader('Compte rendu — opérations de session')

        def _generate_report(ds_name: str, pipeline_ops: list, orig_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> str:
            now = datetime.now().isoformat(sep=' ', timespec='seconds')
            lines = []
            lines.append(f"# Compte rendu — {ds_name}")
            lines.append(f"Date: {now}")
            lines.append("")
            lines.append("## Résumé")
            lines.append(f"- Jeu: {ds_name}")
            lines.append(f"- Opérations en session: {len(pipeline_ops)}")
            lines.append(f"- Forme originale: {orig_df.shape[0]} lignes × {orig_df.shape[1]} colonnes")
            lines.append(f"- Forme actuelle: {cleaned_df.shape[0]} lignes × {cleaned_df.shape[1]} colonnes")
            # nombre de lignes supprimées (positif si on a perdu des lignes)
            try:
                removed_rows = int(orig_df.shape[0] - cleaned_df.shape[0])
            except Exception:
                removed_rows = 'N/A'
            lines.append(f"- Lignes supprimées: {removed_rows}")
            missing_orig = int(orig_df.isna().sum().sum())
            missing_new = int(cleaned_df.isna().sum().sum())
            lines.append(f"- Valeurs manquantes (avant): {missing_orig}")
            lines.append(f"- Valeurs manquantes (après): {missing_new}")
            removed = [c for c in orig_df.columns if c not in cleaned_df.columns]
            lines.append(f"- Colonnes supprimées ({len(removed)}): {', '.join(removed) if removed else 'Aucune'}")
            lines.append("")
            lines.append("## Détail des opérations")
            if not pipeline_ops:
                lines.append("Aucune opération enregistrée.")
            else:
                for i, op in enumerate(pipeline_ops, 1):
                    typ = op.get('op')
                    if typ == 'drop':
                        cols = op.get('cols', [])
                        lines.append(f"{i}. Suppression — colonnes: {', '.join(cols)}")
                    elif typ == 'impute':
                        lines.append(f"{i}. Imputation — colonne: {op.get('col')} — valeur: {op.get('val')}")
                    elif typ == 'cast':
                        lines.append(f"{i}. Typage — colonnes: {', '.join(op.get('cols', []))} → {op.get('dtype')}")
                    elif typ == 'scale':
                        lines.append(f"{i}. Normalisation — colonnes: {', '.join(op.get('cols', []))} — méthode: {op.get('method')}")
                    elif typ == 'drop_rows':
                        idxs = op.get('index', [])
                        # present indices as comma-separated
                        try:
                            ids = ', '.join([str(x) for x in idxs]) if idxs else 'Aucune'
                        except Exception:
                            ids = str(idxs)
                        lines.append(f"{i}. Suppression — lignes (index): {ids}")
                    elif typ == 'drop_rows_cond':
                        cond = op.get('condition', '')
                        lines.append(f"{i}. Suppression — condition: {cond}")
                    elif typ == 'drop_rows_where_na':
                        col = op.get('col', '')
                        lines.append(f"{i}. Suppression — lignes où la colonne '{col}' est manquante")
                    elif typ == 'drop_duplicates':
                        subset = op.get('subset', [])
                        keep = op.get('keep', 'first')
                        subset_txt = ', '.join(subset) if subset else 'toutes les colonnes'
                        if keep is False:
                            keep_txt = 'supprimer toutes les lignes dupliquées'
                        else:
                            keep_txt = f"garder: {keep}"
                        lines.append(f"{i}. Suppression de doublons — colonnes: {subset_txt} — action: {keep_txt}")
                    elif typ == 'set_value':
                        idx = op.get('index')
                        col = op.get('col')
                        val = op.get('val')
                        lines.append(f"{i}. Modification — index: {idx} — colonne: {col} — valeur: {val}")
                    else:
                        # fallback: try to present dict keys in French-friendly way
                        try:
                            lines.append(f"{i}. Opération: {json.dumps(op, default=str, ensure_ascii=False)}")
                        except Exception:
                            lines.append(f"{i}. {op}")
            lines.append("")
            lines.append("## Pipeline (JSON)")
            lines.append("```json")
            lines.append(json.dumps(pipeline_ops, indent=2, default=str))
            lines.append("```")
            return "\n".join(lines)

        orig_df = df_a if ds_clean == a_path.name else df_b
        cleaned_df = st.session_state['df_a_clean'] if ds_clean == a_path.name else st.session_state['df_b_clean']
        pipeline_ops = pipeline or []
        report_md = _generate_report(ds_clean, pipeline_ops, orig_df, cleaned_df)

        if st.button('Générer compte rendu (aperçu)'):
            with st.expander('Aperçu du compte rendu', expanded=True):
                st.markdown(report_md)

        # Téléchargements: markdown et pipeline JSON
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        st.download_button('Télécharger compte rendu (Markdown)', report_md, file_name=f'compte_rendu_{ds_clean}_{timestamp}.md', mime='text/markdown')
        st.download_button('Télécharger pipeline (JSON)', json.dumps(pipeline_ops, indent=2, default=str), file_name=f'pipeline_{ds_clean}_{timestamp}.json', mime='application/json')

    with tab_compare:
        # compute simple scores
        score_a = 0.0
        score_b = 0.0
        tgt_a = detect_target(df_a) is not None
        tgt_b = detect_target(df_b) is not None
        if tgt_a:
            score_a += w_target
        if tgt_b:
            score_b += w_target
        rows_max = max(info_a['rows'], info_b['rows'])
        score_a += w_rows * (info_a['rows'] / rows_max)
        score_b += w_rows * (info_b['rows'] / rows_max)
        miss_max_pct = max(info_a['missing_perc'], info_b['missing_perc'], 1.0)
        out_max_pct = max(info_a.get('outlier_perc', 0.0), info_b.get('outlier_perc', 0.0), 1.0)
        score_a += w_missing * (1 - info_a['missing_perc'] / miss_max_pct)
        score_b += w_missing * (1 - info_b['missing_perc'] / miss_max_pct)
        score_a += w_outlier * (1 - info_a.get('outlier_perc', 0.0) / out_max_pct)
        score_b += w_outlier * (1 - info_b.get('outlier_perc', 0.0) / out_max_pct)

        st.subheader('Recommandation & Scores')
        winner = a_path.name if score_a >= score_b else b_path.name
        st.markdown(f"### ✅ Recommandation: **{winner}**")
        max_score = max(score_a, score_b, 1e-6)
        na = score_a / max_score
        nb = score_b / max_score
        def horizontal_bar(pct: float, color: str, label: str) -> str:
            w = int(pct * 300)
            return f"<div style='background:#e6e6e6;padding:6px;border-radius:6px;width:320px'><div style='width:{w}px;background:{color};height:18px;border-radius:4px'></div></div> <div style='font-size:12px;margin-top:4px'>{label} ({pct*100:.0f}%)</div>"
        def _interp_color(pct: float, start_hex: str, end_hex: str) -> str:
            try:
                pct = max(0.0, min(1.0, float(pct)))
            except Exception:
                pct = 0.0
            def hex_to_rgb(h: str):
                h = h.lstrip('#')
                return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
            def rgb_to_hex(rgb):
                return '#{:02x}{:02x}{:02x}'.format(*[int(max(0, min(255, round(c)))) for c in rgb])
            s = hex_to_rgb(start_hex)
            e = hex_to_rgb(end_hex)
            rgb = (s[0] + (e[0]-s[0]) * pct, s[1] + (e[1]-s[1]) * pct, s[2] + (e[2]-s[2]) * pct)
            return rgb_to_hex(rgb)
        if palette == 'Vert / Rouge':
            low_col, high_col = '#dc2626', '#16a34a'
        else:
            low_col, high_col = '#f97316', '#2563eb'
        colorA = _interp_color(na, low_col, high_col)
        colorB = _interp_color(nb, low_col, high_col)
        colA, colB = st.columns(2)
        with colA:
            st.markdown(horizontal_bar(na, colorA, f"{a_path.name} — {score_a:.2f}"), unsafe_allow_html=True)
        with colB:
            st.markdown(horizontal_bar(nb, colorB, f"{b_path.name} — {score_b:.2f}"), unsafe_allow_html=True)

        st.markdown('---')
        st.write('Comparaison des versions nettoyées disponibles :')
        cleaned_a = st.session_state.get('df_a_clean')
        cleaned_b = st.session_state.get('df_b_clean')
        # Badges par jeu indiquant si la version nettoyée est disponible / contient des opérations
        pa = st.session_state.get('pipeline_a', [])
        pb = st.session_state.get('pipeline_b', [])
        label_a = 'nettoyé' if pa and len(pa) > 0 else 'original / nettoyé (sans ops)'
        label_b = 'nettoyé' if pb and len(pb) > 0 else 'original / nettoyé (sans ops)'
        st.write(f"{a_path.name} — {label_a} — lignes: {len(cleaned_a)}, colonnes: {cleaned_a.shape[1]}")
        st.write(f"{b_path.name} — {label_b} — lignes: {len(cleaned_b)}, colonnes: {cleaned_b.shape[1]}")


if __name__ == '__main__':
    main()
