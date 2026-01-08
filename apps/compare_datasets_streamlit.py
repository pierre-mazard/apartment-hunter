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
import os
import uuid
import shutil
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'
RAW_DIR = DATA_DIR / 'raw'
PROCESSING_DIR = DATA_DIR / 'processing'
CLEANED_DIR = DATA_DIR / 'cleaned'


def ensure_data_dirs() -> None:
    """Create data subdirectories if they do not exist."""
    for d in (DATA_DIR, RAW_DIR, PROCESSING_DIR, CLEANED_DIR):
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass


def _atomic_write_json(path: Path, obj: object) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    try:
        with open(tmp, 'w', encoding='utf8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(str(tmp), str(path))
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass


def _atomic_write_csv(path: Path, df: pd.DataFrame) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    try:
        df.to_csv(tmp, index=False)
        os.replace(str(tmp), str(path))
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass


def _make_processing_names(raw_name: str) -> tuple[Path, Path]:
    stem = Path(raw_name).stem
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    uid = uuid.uuid4().hex[:8]
    proc_name = f"{stem}.{ts}.{uid}.processing.csv"
    pipe_name = f"{stem}.{ts}.{uid}.pipeline.json"
    return (PROCESSING_DIR / proc_name, PROCESSING_DIR / pipe_name)


def save_pipeline_json(pipeline_path: Path, pipeline: list[dict]) -> None:
    try:
        _atomic_write_json(pipeline_path, pipeline)
    except Exception:
        try:
            with open(pipeline_path, 'w', encoding='utf8') as f:
                json.dump(pipeline, f, ensure_ascii=False, indent=2)
        except Exception:
            pass


def load_pipeline_json(pipeline_path: Path) -> list:
    try:
        if pipeline_path.exists():
            with open(pipeline_path, 'r', encoding='utf8') as f:
                return json.load(f)
    except Exception:
        return []
    return []


def create_processing_session_from_raw(raw_name: str, side: str = 'a') -> dict | None:
    """Create a processing CSV and pipeline JSON for a given raw file and attach to session state."""
    try:
        raw_path = RAW_DIR / raw_name
        if not raw_path.exists():
            st.error(f"Raw file not found: {raw_name}")
            return None
        proc_path, pipe_path = _make_processing_names(raw_name)
        df = pd.read_csv(raw_path)
        _atomic_write_csv(proc_path, df)
        # store metadata as first entry in pipeline json for traceability
        save_pipeline_json(pipe_path, [{'meta': {'raw': raw_name, 'proc': proc_path.name, 'created_at': datetime.now().isoformat()}}])
        meta = {'raw': raw_name, 'proc': proc_path.name, 'pipeline': pipe_path.name, 'created_at': datetime.now().isoformat()}
        if side == 'a':
            st.session_state['current_processing_a'] = proc_path.name
            st.session_state['current_pipeline_a'] = pipe_path.name
            st.session_state['current_raw_a'] = raw_name
            st.session_state['pipeline_a'] = []
            st.session_state['df_a_clean'] = df.copy()
        else:
            st.session_state['current_processing_b'] = proc_path.name
            st.session_state['current_pipeline_b'] = pipe_path.name
            st.session_state['current_raw_b'] = raw_name
            st.session_state['pipeline_b'] = []
            st.session_state['df_b_clean'] = df.copy()
        st.success(f"Session de traitement créée: {proc_path.name}")
        return meta
    except Exception as e:
        st.error(f"Erreur création session de traitement: {e}")
        return None


def _make_meta_metrics(orig_df: pd.DataFrame, cleaned_df: pd.DataFrame, pipeline_ops: list[dict]) -> dict:
    """Compute summary metrics to attach in meta."""
    try:
        missing_orig = int(orig_df.isna().sum().sum())
    except Exception:
        missing_orig = None
    try:
        missing_new = int(cleaned_df.isna().sum().sum())
    except Exception:
        missing_new = None
    try:
        cols_removed = [c for c in orig_df.columns if c not in cleaned_df.columns]
    except Exception:
        cols_removed = []
    try:
        cols_added = [c for c in cleaned_df.columns if c not in orig_df.columns]
    except Exception:
        cols_added = []
    try:
        rows_removed_total = int(orig_df.shape[0] - cleaned_df.shape[0])
    except Exception:
        rows_removed_total = None
    ops_count = len([op for op in pipeline_ops if op.get('op')])
    op_summaries: list[str] = []
    for op in pipeline_ops:
        if not op.get('op'):
            continue
        desc = _describe_op(op)
        if op.get('rows_removed') is not None:
            desc = f"{desc} — lignes supprimées: {op.get('rows_removed')}"
        op_summaries.append(desc)
    return {
        'rows_original': int(orig_df.shape[0]) if isinstance(orig_df, pd.DataFrame) else None,
        'cols_original': int(orig_df.shape[1]) if isinstance(orig_df, pd.DataFrame) else None,
        'rows_final': int(cleaned_df.shape[0]) if isinstance(cleaned_df, pd.DataFrame) else None,
        'cols_final': int(cleaned_df.shape[1]) if isinstance(cleaned_df, pd.DataFrame) else None,
        'rows_removed_total': rows_removed_total,
        'missing_before': missing_orig,
        'missing_after': missing_new,
        'cols_removed': cols_removed,
        'cols_added': cols_added,
        'ops_count': ops_count,
        'op_summaries': op_summaries,
    }


def _pipeline_with_meta(raw_name: str, proc_name: str, pipeline_ops: list[dict], meta_extra: dict | None = None) -> list[dict]:
    """Prepend/merge a metadata block for pipeline persistence without polluting op list."""
    meta_block = {'meta': {'raw': raw_name, 'proc': proc_name, 'updated_at': datetime.now().isoformat()}}
    if meta_extra:
        meta_block['meta'].update(meta_extra)
    if pipeline_ops and isinstance(pipeline_ops[0], dict) and 'meta' in pipeline_ops[0]:
        new_pipe = pipeline_ops.copy()
        meta = new_pipe[0].get('meta', {}) or {}
        meta.update(meta_block['meta'])
        new_pipe[0]['meta'] = meta
        return new_pipe
    return [meta_block] + pipeline_ops


def _reapply_pipeline_and_persist(side: str) -> None:
    """Reapply pipeline to raw and persist processing CSV + pipeline JSON."""
    try:
        side_key = 'a' if side == 'a' else 'b'
        if side == 'a':
            pipeline = st.session_state.get('pipeline_a', [])
            proc_name = st.session_state.get('current_processing_a')
            pipe_name = st.session_state.get('current_pipeline_a')
            raw_name = st.session_state.get('current_raw_a') or st.session_state.get('name_a')
        else:
            pipeline = st.session_state.get('pipeline_b', [])
            proc_name = st.session_state.get('current_processing_b')
            pipe_name = st.session_state.get('current_pipeline_b')
            raw_name = st.session_state.get('current_raw_b') or st.session_state.get('name_b')
        # Fallbacks: if pipeline name missing but processing present, try to recover/seed it
        if proc_name and not pipe_name:
            pipeline_guess = proc_name.replace('.processing.csv', '.pipeline.json')
            pj_path = PROCESSING_DIR / pipeline_guess
            if pj_path.exists():
                pipe_name = pipeline_guess
                pipeline = load_pipeline_json(pj_path)
            else:
                guessed_raw = raw_name or f"{Path(proc_name).stem.split('.')[0]}.csv"
                pipe_name = pipeline_guess
                pipeline = [{'meta': {'raw': guessed_raw, 'proc': proc_name, 'created_at': datetime.now().isoformat()}}]
            st.session_state[f'current_pipeline_{side_key}'] = pipe_name
            st.session_state[f'pipeline_{side_key}'] = pipeline
        # If raw name missing, derive from pipeline meta or processing filename
        if not raw_name:
            guessed_raw = None
            try:
                if pipeline and isinstance(pipeline[0], dict) and 'meta' in pipeline[0]:
                    guessed_raw = pipeline[0].get('meta', {}).get('raw')
            except Exception:
                guessed_raw = None
            if not guessed_raw and proc_name:
                guessed_raw = f"{Path(proc_name).stem.split('.')[0]}.csv"
            raw_name = guessed_raw
            if raw_name:
                st.session_state[f'current_raw_{side_key}'] = raw_name
        if not proc_name or not pipe_name or not raw_name:
            st.warning("Impossible de persister: session processing ou pipeline manquante.")
            return
        raw_path = RAW_DIR / raw_name
        if not raw_path.exists():
            st.warning(f"Fichier raw introuvable pour persistance: {raw_name}")
            return
        df_raw = pd.read_csv(raw_path)
        ops_enriched = _annotate_pipeline_with_row_diffs(df_raw, [op for op in pipeline if op.get('op')])
        df_proc = apply_pipeline(df_raw, [op for op in ops_enriched if op.get('op')])
        meta_extra = _make_meta_metrics(df_raw, df_proc, ops_enriched)
        proc_path = PROCESSING_DIR / proc_name
        pipe_path = PROCESSING_DIR / pipe_name
        try:
            _atomic_write_csv(proc_path, df_proc)
        except Exception as e:
            st.error(f"Échec écriture processing CSV: {e}")
            return
        try:
            save_pipeline_json(pipe_path, _pipeline_with_meta(raw_name, proc_name, ops_enriched, meta_extra=meta_extra))
        except Exception as e:
            st.error(f"Échec écriture pipeline JSON: {e}")
            return
        if side == 'a':
            st.session_state['df_a_clean'] = df_proc
        else:
            st.session_state['df_b_clean'] = df_proc
    except Exception as e:
        st.error(f"Persistance échouée: {e}")
        return


def add_op(side: str, op: dict) -> None:
    key = 'pipeline_a' if side == 'a' else 'pipeline_b'
    if key not in st.session_state:
        st.session_state[key] = []
    st.session_state[key].append(op)
    _reapply_pipeline_and_persist(side)


def apply_pipeline(original: pd.DataFrame, pipeline: list[dict]) -> pd.DataFrame:
    df = original.copy()

    def _insert_flag(df_local: pd.DataFrame, col: str, flag_col: str, flag_values: pd.Series) -> pd.DataFrame:
        # ensure alignment
        flag_series = pd.Series(flag_values, index=df_local.index)
        df_local[flag_col] = flag_series.astype('Int64')
        cols = list(df_local.columns)
        if flag_col in cols and col in cols:
            cols.remove(flag_col)
            try:
                idx = cols.index(col)
                cols.insert(idx + 1, flag_col)
                df_local = df_local[cols]
            except ValueError:
                pass
        return df_local
    for op in pipeline:
        if not op.get('op'):
            continue
        if op.get('op') == 'drop':
            cols = op.get('cols', [])
            df = df.drop(columns=cols, errors='ignore')
        elif op.get('op') == 'drop_rows':
            idxs = op.get('index', []) or []
            try:
                df = df.drop(index=[i for i in idxs if i in df.index], errors='ignore')
            except Exception:
                try:
                    df = df.drop(index=idxs, errors='ignore')
                except Exception:
                    continue
        elif op.get('op') == 'drop_rows_cond':
            cond = op.get('condition')
            if not cond:
                continue
            try:
                mask = df.query(cond).index
                df = df.drop(index=mask, errors='ignore')
            except Exception:
                continue
        elif op.get('op') == 'drop_rows_where_na':
            col = op.get('col')
            if not col or col not in df.columns:
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
                mask_na = df[col].isna()
                df[col] = df[col].fillna(val)
                flag_col = op.get('flag_col') or f"{col}_imputed_flag"
                df = _insert_flag(df, col, flag_col, mask_na.astype(int))
        elif op.get('op') == 'impute_calc':
            col = op.get('col')
            num = op.get('numerator')
            den = op.get('denominator')
            rounding = (op.get('round') or 'float').lower()
            if col in df.columns and num in df.columns and den in df.columns:
                try:
                    num_ser = pd.to_numeric(df[num], errors='coerce')
                    den_ser = pd.to_numeric(df[den], errors='coerce')
                    calc = num_ser / den_ser
                    if rounding == 'int':
                        # round and cast to pandas nullable Int64 to preserve NA
                        calc = calc.round().astype('Int64')
                    else:
                        # ensure float with reasonable precision
                        calc = calc.astype(float)
                    # fill only where target is NA and calc is not NA
                    tgt = df[col]
                    mask_na = tgt.isna() & calc.notna()
                    df.loc[mask_na, col] = calc[mask_na]
                    flag_col = op.get('flag_col') or f"{col}_imputed_flag"
                    df = _insert_flag(df, col, flag_col, mask_na.astype(int))
                except Exception:
                    pass
        elif op.get('op') == 'cast':
            cols = op.get('cols', [])
            dtype = op.get('dtype')
            for col in cols:
                if col not in df.columns:
                    continue
                try:
                    if dtype == 'int':
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
                    continue
        elif op.get('op') == 'scale':
            cols = op.get('cols', [])
            method = op.get('method')
            params = op.get('params', {}) or {}
            for col in cols:
                if col not in df.columns:
                    continue
                try:
                    ser = pd.to_numeric(df[col], errors='coerce')
                    if method == 'standard':
                        mean = ser.mean()
                        std = ser.std() or 1.0
                        df[col] = (ser - mean) / std
                    elif method == 'minmax':
                        min_v = ser.min()
                        max_v = ser.max()
                        if max_v == min_v:
                            continue
                        df[col] = (ser - min_v) / (max_v - min_v)
                    elif method == 'robust':
                        q1 = ser.quantile(0.25)
                        q3 = ser.quantile(0.75)
                        iqr = q3 - q1 or 1.0
                        df[col] = (ser - q1) / iqr
                except Exception:
                    continue
        elif op.get('op') == 'fillna_mode':
            cols = op.get('cols', [])
            groupby_col = op.get('groupby')
            for col in cols:
                if col not in df.columns:
                    continue
                try:
                    if groupby_col and groupby_col in df.columns:
                        mask_na = df[col].isna()
                        grp = df.groupby(groupby_col)[col].agg(lambda s: s.mode(dropna=True).iloc[0] if not s.mode(dropna=True).empty else None)
                        vals = df.loc[mask_na, groupby_col].map(grp)
                        df.loc[mask_na, col] = vals
                        flag_col = op.get('flag_col') or f"{col}_imputed_flag"
                        df = _insert_flag(df, col, flag_col, mask_na.astype(int))
                    else:
                        mode = df[col].mode(dropna=True)
                        if not mode.empty:
                            mask_na = df[col].isna()
                            df[col] = df[col].fillna(mode.iloc[0])
                            flag_col = op.get('flag_col') or f"{col}_imputed_flag"
                            df = _insert_flag(df, col, flag_col, mask_na.astype(int))
                except Exception:
                    continue
        elif op.get('op') == 'impute_mode_multi_group':
            col = op.get('col')
            groupby_cols = op.get('groupby_cols', [])
            if col in df.columns and groupby_cols:
                try:
                    mask_na = df[col].isna()
                    if mask_na.any() and all(c in df.columns for c in groupby_cols):
                        grp = df.groupby(groupby_cols)[col].agg(lambda s: s.mode(dropna=True).iloc[0] if not s.mode(dropna=True).empty else None)
                        vals = df.loc[mask_na, groupby_cols].apply(lambda row: grp.get(tuple(row), None), axis=1)
                        df.loc[mask_na, col] = vals
                        flag_col = op.get('flag_col') or f"{col}_imputed_flag"
                        df = _insert_flag(df, col, flag_col, mask_na.astype(int))
                except Exception:
                    continue
        elif op.get('op') == 'impute_inverse_bool':
            col_target = op.get('col_target')
            col_source = op.get('col_source')
            if col_target in df.columns and col_source in df.columns:
                try:
                    # Transformer TOUTES les valeurs avec l'inverse booléen de col_source
                    inverted = ~df[col_source].astype(bool)
                    df[col_target] = inverted
                    flag_col = op.get('flag_col') or f"{col_target}_imputed_flag"
                    # Le flag indique que toutes les valeurs ont été transformées
                    df = _insert_flag(df, col_target, flag_col, pd.Series([1] * len(df), index=df.index))
                except Exception:
                    continue
        elif op.get('op') == 'merge_bool_to_category':
            col_central = op.get('col_central')
            col_individual = op.get('col_individual')
            col_target = op.get('col_target')
            drop_sources = op.get('drop_sources', True)
            if col_central in df.columns and col_individual in df.columns:
                try:
                    # Créer la nouvelle colonne catégorielle
                    df[col_target] = 'unknown'
                    df.loc[df[col_central] == True, col_target] = 'central'
                    df.loc[df[col_individual] == True, col_target] = 'individual'
                    # Marquer comme NA si les deux colonnes sources sont NA
                    mask_both_na = df[col_central].isna() & df[col_individual].isna()
                    df.loc[mask_both_na, col_target] = pd.NA
                    # Supprimer les colonnes sources si demandé
                    if drop_sources:
                        df = df.drop(columns=[col_central, col_individual])
                except Exception:
                    continue
        elif op.get('op') == 'rename':
            mapping = op.get('mapping', {})
            try:
                df = df.rename(columns=mapping)
            except Exception:
                continue
        elif op.get('op') == 'keep_cols':
            cols = op.get('cols', [])
            try:
                df = df[cols]
            except Exception:
                continue
        elif op.get('op') == 'drop_duplicates':
            subset = op.get('subset') or None
            keep = op.get('keep', 'first')
            try:
                df = df.drop_duplicates(subset=subset, keep=keep)
            except Exception:
                continue
        elif op.get('op') == 'clip':
            col = op.get('col')
            lower = op.get('lower')
            upper = op.get('upper')
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce').clip(lower=lower, upper=upper)
                except Exception:
                    continue
        elif op.get('op') == 'fill_const':
            col = op.get('col')
            val = op.get('val')
            if col in df.columns:
                df[col] = df[col].fillna(val)
        elif op.get('op') == 'concat':
            cols = op.get('cols', [])
            dest = op.get('dest')
            sep = op.get('sep', ' ')
            if dest and cols:
                try:
                    df[dest] = df[cols].astype(str).agg(sep.join, axis=1)
                except Exception:
                    continue
        elif op.get('op') == 'lower':
            cols = op.get('cols', [])
            for col in cols:
                if col in df.columns:
                    try:
                        df[col] = df[col].astype('string').str.lower()
                    except Exception:
                        continue
        elif op.get('op') == 'strip':
            cols = op.get('cols', [])
            for col in cols:
                if col in df.columns:
                    try:
                        df[col] = df[col].astype('string').str.strip()
                    except Exception:
                        continue
        elif op.get('op') == 'remove_text':
            col = op.get('col')
            text = op.get('text', '')
            if col in df.columns and text:
                try:
                    df[col] = df[col].astype('string').str.replace(text, '', regex=False).str.strip()
                except Exception:
                    continue
        elif op.get('op') == 'split_neighborhood':
            col = op.get('col')
            code_col = op.get('code_col') or 'neighborhood_code'
            name_col = op.get('name_col') or 'neighborhood_name'
            price_col = op.get('price_col') or 'neighborhood_price_sqm'
            dist_code_col = op.get('dist_code_col') or 'district_code'
            dist_name_col = op.get('dist_name_col') or 'district_name'
            drop_source = op.get('drop_source', False)
            if col in df.columns:
                try:
                    pattern = r"Neighborhood\s+(\d+):\s*(.*?)\s*\(([\d\.]+)\s*€/m2\)\s*-\s*District\s+(\d+):\s*(.*)"
                    extracted = df[col].astype('string').str.extract(pattern)
                    if extracted.shape[1] == 5:
                        df[code_col] = extracted[0].astype('Int64')
                        df[name_col] = extracted[1].astype('string').str.strip()
                        df[price_col] = pd.to_numeric(extracted[2], errors='coerce')
                        df[dist_code_col] = extracted[3].astype('Int64')
                        df[dist_name_col] = extracted[4].astype('string').str.strip()
                        if drop_source:
                            df = df.drop(columns=[col], errors='ignore')
                except Exception:
                    continue
        elif op.get('op') == 'split_category':
            col = op.get('col')
            pattern = op.get('pattern', ':')  # séparateur ou regex
            col_id = op.get('col_id')
            col_desc = op.get('col_desc')
            drop_source = op.get('drop_source', False)
            if col in df.columns and col_id and col_desc:
                try:
                    # Split sur le pattern
                    split_result = df[col].astype('string').str.split(pattern, n=1, expand=True)
                    if split_result.shape[1] >= 2:
                        # Première partie : extraire le numéro
                        df[col_id] = split_result[0].str.extract(r'(\d+)', expand=False).astype('Int64')
                        # Deuxième partie : la description (strip les espaces)
                        df[col_desc] = split_result[1].str.strip()
                        if drop_source:
                            df = df.drop(columns=[col], errors='ignore')
                except Exception:
                    continue
        elif op.get('op') == 'impute_extract_text':
            col_source = op.get('col_source')
            col_target = op.get('col_target')
            extract_pattern = op.get('extract_pattern', '')
            if col_source in df.columns and col_target in df.columns and extract_pattern:
                try:
                    mask_na = df[col_target].isna()
                    if mask_na.any():
                        # Extraire le texte après le pattern dans col_source
                        extracted = df[col_source].astype('string').str.split(extract_pattern, n=1, expand=True)
                        if extracted.shape[1] >= 2:
                            extracted_text = extracted[1].str.strip()
                            # Imputer uniquement les NA dans col_target
                            df.loc[mask_na, col_target] = extracted_text[mask_na]
                            # Ajouter flag d'imputation
                            flag_col = op.get('flag_col') or f"{col_target}_imputed_flag"
                            df = _insert_flag(df, col_target, flag_col, mask_na.astype(int))
                except Exception:
                    continue
        elif op.get('op') == 'surface_coherence_flag':
            tol = op.get('tolerance_pct', 20)
            required = ['sq_mt_built', 'buy_price', 'buy_price_by_area']
            if all(c in df.columns for c in required):
                try:
                    calc = (pd.to_numeric(df['buy_price'], errors='coerce') / pd.to_numeric(df['buy_price_by_area'], errors='coerce'))
                    built = pd.to_numeric(df['sq_mt_built'], errors='coerce')
                    div_pct = (abs(built - calc) / built * 100)
                    df['surface_calc_from_price'] = calc.round(2)
                    df['surface_coherence_divergence_pct'] = div_pct.round(2)
                    df['surface_incoherent_flag'] = (div_pct > tol)
                except Exception:
                    pass
    return df


def _annotate_pipeline_with_row_diffs(orig_df: pd.DataFrame, pipeline_ops: list[dict]) -> list[dict]:
    """Return a copy of pipeline_ops with 'rows_removed' per step (best-effort)."""
    enriched: list[dict] = []
    df_prev = orig_df.copy()
    row_ops = {'drop_rows', 'drop_rows_cond', 'drop_rows_where_na', 'drop_duplicates'}
    for op in pipeline_ops:
        op_copy = op.copy()
        if not op_copy.get('op'):
            enriched.append(op_copy)
            continue
        try:
            df_next = apply_pipeline(df_prev, [op_copy])
            if op_copy.get('op') in row_ops:
                op_copy['rows_removed'] = int(df_prev.shape[0] - df_next.shape[0])
            else:
                op_copy.pop('rows_removed', None)
            df_prev = df_next
        except Exception:
            if op_copy.get('op') in row_ops:
                op_copy['rows_removed'] = None
            else:
                op_copy.pop('rows_removed', None)
        enriched.append(op_copy)
    return enriched


def _describe_op(op: dict) -> str:
    """Human-friendly description for an op (used in meta / report)."""
    typ = op.get('op')
    if not typ:
        return json.dumps(op, ensure_ascii=False)
    if typ == 'drop':
        cols = op.get('cols', [])
        reason = op.get('reason')
        reason_custom = op.get('reason_text')
        reason_txt = ''
        if reason == 'manual_selection':
            reason_txt = 'sélection manuelle'
            if reason_custom:
                reason_txt = f"{reason_txt} — {reason_custom}"
        elif reason == 'auto_empty_cols_0':
            reason_txt = 'colonnes vides (0%)'
        elif reason == 'auto_full_cols_100':
            reason_txt = 'colonnes complètes (100%)'
        prefix = 'Suppression de colonnes'
        if reason_txt:
            prefix = f"{prefix} — {reason_txt}"
        return f"{prefix} ({len(cols)}): {', '.join(cols)}"
    if typ == 'drop_rows':
        idxs = op.get('index', [])
        return f"Suppression de lignes (index explicites): {idxs}"
    if typ == 'drop_rows_cond':
        cond = op.get('condition', '')
        if op.get('coherence_check'):
            coherence_type = op.get('coherence_type', 'validation')
            reason = op.get('reason_text', 'validation croisée')
            if coherence_type == 'surface_validation':
                return f"Suppression de lignes (incohérence superficie: {reason})"
            return f"Suppression de lignes (incohérence: {reason}): {cond}"
        return f"Suppression de lignes (condition): {cond}"
    if typ == 'drop_rows_where_na':
        col = op.get('col', '')
        return f"Suppression de lignes (NA dans {col})"
    if typ == 'impute_calc':
        col = op.get('col', '')
        num = op.get('numerator', '')
        den = op.get('denominator', '')
        rounding = (op.get('round') or 'float')
        return f"Imputation calculée — {col} = {num} / {den} ({rounding})"
    if typ == 'surface_coherence_flag':
        tol = op.get('tolerance_pct', 20)
        return f"Marquage incohérences superficie (tolérance {tol}%)"
    if typ == 'drop_duplicates':
        subset = op.get('subset', []) or 'toutes les colonnes'
        keep = op.get('keep', 'first')
        return f"Suppression de doublons — colonnes: {subset} — action: garder {keep}"
    if typ == 'impute':
        return f"Imputation — {op.get('col')} = {op.get('val')}"
    if typ == 'fillna_mode':
        cols = op.get('cols', [])
        groupby_col = op.get('groupby')
        if groupby_col:
            return f"Imputation — mode par groupe — {', '.join(cols)} groupé par {groupby_col}"
        else:
            return f"Imputation — mode global — {', '.join(cols)}"
    if typ == 'impute_mode_multi_group':
        col = op.get('col', '')
        groupby_cols = op.get('groupby_cols', [])
        cols_str = ', '.join(groupby_cols) if groupby_cols else 'groupes'
        return f"Imputation — {col} par mode/médiane groupé par: {cols_str}"
    if typ == 'impute_inverse_bool':
        col_target = op.get('col_target', '')
        col_source = op.get('col_source', '')
        return f"Imputation logique — {col_target} = NOT {col_source}"
    if typ == 'merge_bool_to_category':
        col_central = op.get('col_central', '')
        col_individual = op.get('col_individual', '')
        col_target = op.get('col_target', '')
        drop = op.get('drop_sources', True)
        action = "fusion + suppression" if drop else "fusion"
        return f"Fusion booléenne — {col_central} + {col_individual} → {col_target} ({action})"
    if typ == 'impute_extract_text':
        col_source = op.get('col_source', '')
        col_target = op.get('col_target', '')
        pattern = op.get('extract_pattern', '')
        return f"Imputation par extraction — {col_target} extrait de {col_source} (après '{pattern}')"
    if typ == 'remove_text':
        return f"Nettoyage texte — suppression de '{op.get('text', '')}' dans {op.get('col', '')}"
    if typ == 'split_neighborhood':
        col = op.get('col', '')
        code_col = op.get('code_col', 'neighborhood_code')
        name_col = op.get('name_col', 'neighborhood_name')
        price_col = op.get('price_col', 'neighborhood_price_sqm')
        dist_code_col = op.get('dist_code_col', 'district_code')
        dist_name_col = op.get('dist_name_col', 'district_name')
        drop_src = op.get('drop_source', False)
        drop_txt = " (suppression colonne source)" if drop_src else ""
        return f"Scission quartier — {col} → {code_col}, {name_col}, {price_col}, {dist_code_col}, {dist_name_col}{drop_txt}"
    if typ == 'split_category':
        col = op.get('col', '')
        col_id = op.get('col_id', '')
        col_desc = op.get('col_desc', '')
        drop_src = op.get('drop_source', False)
        drop_txt = " (suppression colonne source)" if drop_src else ""
        return f"Scission de colonne — {col} → {col_id}, {col_desc}{drop_txt}"
    if typ == 'cast':
        cols = op.get('cols', [])
        dtype = op.get('dtype')
        prev = op.get('dtype_prev') or {}
        if isinstance(prev, dict) and prev:
            parts = []
            for col in cols:
                prev_t = prev.get(col)
                if prev_t:
                    parts.append(f"{col}: {prev_t} → {dtype}")
                else:
                    parts.append(f"{col} → {dtype}")
            return f"Typage — {'; '.join(parts)}"
        return f"Typage — {', '.join(cols)} → {dtype}"
    if typ == 'scale':
        return f"Normalisation — {', '.join(op.get('cols', []))} — méthode {op.get('method')}"
    if typ == 'set_value':
        return f"Modification — idx {op.get('index')} col {op.get('col')} → {op.get('val')}"
    return json.dumps(op, ensure_ascii=False)


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

    # ensure expected data subfolders exist
    ensure_data_dirs()

    a_path = RAW_DIR / 'houses_Madrid.csv'
    b_path = RAW_DIR / 'kc_house_data.csv'

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
    tab_load, tab_diag, tab_explore, tab_clean, tab_compare = st.tabs(['Chargement des jeux de données', 'Diagnostic — Qualité', 'Exploration des données', 'Nettoyage — Préparer', 'Comparaison'])

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
        use_clean = st.checkbox('Afficher la version nettoyée en session (si disponible)', value=True)
        if use_clean:
            # préférer la version en session ; si une session processing est active, relire le fichier processing pour rester synchro
            df = st.session_state.get('df_a_clean') if dataset == a_path.name else st.session_state.get('df_b_clean')
            proc_name = st.session_state.get('current_processing_a') if dataset == a_path.name else st.session_state.get('current_processing_b')
            if proc_name:
                try:
                    df = pd.read_csv(PROCESSING_DIR / proc_name)
                except Exception:
                    pass
            # fallback to original if rien n'est disponible
            if df is None:
                df = df_a if dataset == a_path.name else df_b
        else:
            df = df_a if dataset == a_path.name else df_b
        # badge restera géré plus bas (section provenance)

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
        # list files present in raw / processing / cleaned
        try:
            raw_files = sorted([p.name for p in RAW_DIR.glob('*.csv')])
        except Exception:
            raw_files = []
        try:
            processing_files = sorted([p.name for p in PROCESSING_DIR.glob('*.csv')])
        except Exception:
            processing_files = []
        try:
            cleaned_files_list = sorted([p.name for p in CLEANED_DIR.glob('*.csv')])
        except Exception:
            cleaned_files_list = []

        # display counts and short previews
        st.subheader('Fichiers disponibles')
        st.write(f'- raw: {len(raw_files)}')
        if raw_files:
            st.write(', '.join(raw_files[:50]))
        st.write(f'- processing: {len(processing_files)}')
        if processing_files:
            st.write(', '.join(processing_files[:50]))
        st.write(f'- cleaned: {len(cleaned_files_list)}')
        if cleaned_files_list:
            st.write(', '.join(cleaned_files_list[:50]))

        # multi-select: choose 1..N raw files to work with
        sel_raw_multi = st.multiselect('Sélectionner un ou plusieurs jeux bruts (data/raw)', options=raw_files, key='sel_raw_multi')
        st.session_state['selected_raw_files'] = sel_raw_multi

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader('Jeu A')
            sel_a = st.selectbox('Choisir un CSV existant (data/raw)', [''] + raw_files, key='sel_a_file')
            up_a = st.file_uploader('Ou téléverser un CSV pour A', type=['csv'], key='upload_a')
            if sel_a:
                path_a = RAW_DIR / sel_a
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
                    # avoid re-saving the same uploaded file on every rerun
                    raw_orig_name = getattr(up_a, 'name', 'upload_a.csv')
                    raw_bytes = up_a.getvalue()
                    raw_size = len(raw_bytes)
                    already_saved = st.session_state.get('uploaded_a_saved', False)
                    saved_name = st.session_state.get('uploaded_a_saved_name')
                    saved_size = st.session_state.get('uploaded_a_saved_size')
                    if already_saved and saved_name == raw_orig_name and saved_size == raw_size:
                        # already saved this upload in this session — just load the saved file
                        dest = RAW_DIR / saved_name
                        if dest.exists():
                            df_new = pd.read_csv(dest)
                            st.session_state['df_a_custom'] = df_new
                            st.session_state['name_a'] = dest.name
                            st.session_state['df_a_clean'] = df_new.copy()
                            st.session_state['pipeline_a'] = []
                    else:
                        # save uploaded raw CSV into data/raw/ preserving filename
                        fname = raw_orig_name
                        dest = RAW_DIR / fname
                        if dest.exists():
                            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                            dest = RAW_DIR / f"{dest.stem}.{ts}{dest.suffix}"
                        with open(dest, 'wb') as f:
                            f.write(raw_bytes)
                        # load from saved file to ensure parity
                        df_new = pd.read_csv(dest)
                        st.session_state['df_a_custom'] = df_new
                        st.session_state['name_a'] = dest.name
                        st.session_state['df_a_clean'] = df_new.copy()
                        st.session_state['pipeline_a'] = []
                        st.session_state['uploaded_a_saved'] = True
                        st.session_state['uploaded_a_saved_name'] = dest.name
                        st.session_state['uploaded_a_saved_size'] = raw_size
                        st.success(f'CSV uploadé et enregistré dans `data/raw/{dest.name}` puis chargé pour Jeu A')
                except Exception as e:
                    st.error(f'Echec lecture/écriture du CSV uploadé: {e}')
        with col_b:
            st.subheader('Jeu B')
            sel_b = st.selectbox('Choisir un CSV existant (data/raw)', [''] + raw_files, key='sel_b_file')
            up_b = st.file_uploader('Ou téléverser un CSV pour B', type=['csv'], key='upload_b')
            if sel_b:
                path_b2 = RAW_DIR / sel_b
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
                    # avoid re-saving the same uploaded file on every rerun
                    raw_orig_name = getattr(up_b, 'name', 'upload_b.csv')
                    raw_bytes = up_b.getvalue()
                    raw_size = len(raw_bytes)
                    already_saved_b = st.session_state.get('uploaded_b_saved', False)
                    saved_name_b = st.session_state.get('uploaded_b_saved_name')
                    saved_size_b = st.session_state.get('uploaded_b_saved_size')
                    if already_saved_b and saved_name_b == raw_orig_name and saved_size_b == raw_size:
                        dest = RAW_DIR / saved_name_b
                        if dest.exists():
                            df_new = pd.read_csv(dest)
                            st.session_state['df_b_custom'] = df_new
                            st.session_state['name_b'] = dest.name
                            st.session_state['df_b_clean'] = df_new.copy()
                            st.session_state['pipeline_b'] = []
                    else:
                        # save uploaded raw CSV into data/raw/ preserving filename
                        fname = raw_orig_name
                        dest = RAW_DIR / fname
                        if dest.exists():
                            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                            dest = RAW_DIR / f"{dest.stem}.{ts}{dest.suffix}"
                        with open(dest, 'wb') as f:
                            f.write(raw_bytes)
                        df_new = pd.read_csv(dest)
                        st.session_state['df_b_custom'] = df_new
                        st.session_state['name_b'] = dest.name
                        st.session_state['df_b_clean'] = df_new.copy()
                        st.session_state['pipeline_b'] = []
                        st.session_state['uploaded_b_saved'] = True
                        st.session_state['uploaded_b_saved_name'] = dest.name
                        st.session_state['uploaded_b_saved_size'] = raw_size
                        st.success(f'CSV uploadé et enregistré dans `data/raw/{dest.name}` puis chargé pour Jeu B')
                except Exception as e:
                    st.error(f'Echec lecture/écriture du CSV uploadé: {e}')

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
        # Controls: select dataset side (A/B). The actual source used for cleaning
        # will be the active processing session if present (preferred), otherwise
        # the in-session/raw dataframe.
        ds_clean = st.selectbox('Choisir jeu à nettoyer (côté A/B)', [a_path.name, b_path.name])
        side = 'a' if ds_clean == a_path.name else 'b'
        # if there is a processing session for this side, prefer it by default
        proc_name = st.session_state.get('current_processing_a') if side == 'a' else st.session_state.get('current_processing_b')
        pipe_name = st.session_state.get('current_pipeline_a') if side == 'a' else st.session_state.get('current_pipeline_b')
        use_proc_default = bool(proc_name)
        use_processing = st.checkbox('Utiliser la session de traitement `processing` (si disponible)', value=use_proc_default, key=f'use_processing_{side}')
        # choose the dataframe and pipeline depending on processing session presence
        if use_processing and proc_name:
            try:
                df_clean = pd.read_csv(PROCESSING_DIR / proc_name)
            except Exception:
                df_clean = st.session_state['df_a_clean'] if side == 'a' else st.session_state['df_b_clean']
            pipeline = st.session_state.get('pipeline_a') if side == 'a' else st.session_state.get('pipeline_b')
            is_clean_view = True
            source_label = f"processing/{proc_name} (session active)"
        else:
            df_clean = st.session_state['df_a_clean'] if side == 'a' else st.session_state['df_b_clean']
            pipeline = st.session_state.get('pipeline_a') if side == 'a' else st.session_state.get('pipeline_b')
            is_clean_view = bool(pipeline and len(pipeline) > 0)
            source_label = 'version en session' if is_clean_view else 'version originale (raw)'
        # processing sessions: allow creating a session from a raw file or resuming existing processing files
        try:
            raw_files_local = sorted([p.name for p in RAW_DIR.glob('*.csv')])
        except Exception:
            raw_files_local = []
        try:
            processing_files_local = sorted([p.name for p in PROCESSING_DIR.glob('*.csv')])
        except Exception:
            processing_files_local = []
        with st.expander('Sessions de traitement (processing)', expanded=False):
            col1, col2 = st.columns([2, 1])
            with col1:
                sel_raw_for_session = st.selectbox('Créer une session à partir d\'un raw', [''] + raw_files_local, key='sel_raw_for_session')
            with col2:
                if st.button('Créer session de traitement', key=f'create_proc_{ds_clean}') and st.session_state.get('sel_raw_for_session'):
                    chosen = st.session_state.get('sel_raw_for_session')
                    side = 'a' if ds_clean == a_path.name else 'b'
                    meta = create_processing_session_from_raw(chosen, side=side)
                    if meta:
                        st.session_state['last_created_processing'] = meta
            st.markdown('---')
            st.write('Sessions existantes:')
            if processing_files_local:
                for pfn in processing_files_local:
                    rcol1, rcol2 = st.columns([3, 1])
                    rcol1.write(pfn)
                    if rcol2.button('Reprendre', key=f'resume_{pfn}'):
                        # when resuming, locate associated pipeline JSON
                        st.session_state['current_processing_a' if ds_clean == a_path.name else 'current_processing_b'] = pfn
                        # attempt to find pipeline json with same base (processing filenames end with .processing.csv)
                        pipeline_guess = pfn.replace('.processing.csv', '.pipeline.json')
                        pj_path = PROCESSING_DIR / pipeline_guess
                        if pj_path.exists():
                            pj = pj_path.name
                        else:
                            # fallback glob (older naming quirks)
                            stem = Path(pfn).stem  # e.g., houses_Madrid.20260108_105301.5b301f0c.processing
                            stem_no_processing = stem.replace('.processing', '')
                            candidates = list(PROCESSING_DIR.glob(f"{stem_no_processing}*.pipeline.json")) or list(PROCESSING_DIR.glob(f"{stem}*.pipeline.json"))
                            pj = candidates[0].name if candidates else None
                        # best-effort guess of original raw filename
                        guessed_raw = f"{Path(pfn).stem.split('.')[0]}.csv"
                        if ds_clean == a_path.name:
                            if pj:
                                st.session_state['current_pipeline_a'] = pj
                                st.session_state['pipeline_a'] = load_pipeline_json(PROCESSING_DIR / pj)
                            else:
                                # create a fresh pipeline json name based on processing file and seed with meta
                                st.session_state['current_pipeline_a'] = pipeline_guess
                                st.session_state['pipeline_a'] = [{'meta': {'raw': guessed_raw, 'proc': pfn, 'created_at': datetime.now().isoformat()}}]
                            # load processing CSV snapshot if available
                            try:
                                st.session_state['df_a_clean'] = pd.read_csv(PROCESSING_DIR / pfn)
                            except Exception:
                                pass
                            # current_raw: prefer metadata, fallback to guessed raw name
                            try:
                                st.session_state['current_raw_a'] = st.session_state['pipeline_a'][0].get('raw') if st.session_state['pipeline_a'] else guessed_raw
                            except Exception:
                                st.session_state['current_raw_a'] = guessed_raw
                        else:
                            if pj:
                                st.session_state['current_pipeline_b'] = pj
                                st.session_state['pipeline_b'] = load_pipeline_json(PROCESSING_DIR / pj)
                            else:
                                st.session_state['current_pipeline_b'] = pipeline_guess
                                st.session_state['pipeline_b'] = [{'meta': {'raw': guessed_raw, 'proc': pfn, 'created_at': datetime.now().isoformat()}}]
                            try:
                                st.session_state['df_b_clean'] = pd.read_csv(PROCESSING_DIR / pfn)
                            except Exception:
                                pass
                            try:
                                st.session_state['current_raw_b'] = st.session_state['pipeline_b'][0].get('raw') if st.session_state['pipeline_b'] else guessed_raw
                            except Exception:
                                st.session_state['current_raw_b'] = guessed_raw
                        st.success(f'Repris session: {pfn}')
            else:
                st.write('_Aucune session de traitement trouvée._')
        # Display a badge indicating the current source (processing session preferred)
        try:
            ops_count = len([op for op in (pipeline or []) if op.get('op')])
        except Exception:
            ops_count = 0
        if use_processing and proc_name:
            ptxt = f"Affichage: <b>{source_label}</b> — opérations en session: {ops_count}"
            pcol = '#16a34a'
        else:
            if ops_count > 0:
                ptxt = f"Affichage: <b>{source_label}</b> — opérations en session: {ops_count}"
                pcol = '#16a34a'
            else:
                ptxt = f"Affichage: <b>{source_label}</b>"
                pcol = '#6b7280'
        st.markdown(f"<div style='padding:6px;border-radius:6px;background:{pcol};color:white;display:inline-block'>{ptxt}</div>", unsafe_allow_html=True)

        # Undo / reset controls (operate on pipeline and persisted processing if exists)
        undo_col1, undo_col2 = st.columns([1, 1])
        side_hint = 'a' if ds_clean == a_path.name else 'b'
        with undo_col1:
            if st.button('Annuler dernière opération', key=f'undo_processing_{ds_clean}'):
                side = side_hint
                key = 'pipeline_a' if side == 'a' else 'pipeline_b'
                if st.session_state.get(key):
                    try:
                        st.session_state[key].pop()
                        _reapply_pipeline_and_persist(side)
                        st.success('Dernière opération annulée.')
                    except Exception as e:
                        st.error(f"Impossible d'annuler: {e}")
                else:
                    st.info('Aucune opération à annuler.')
        with undo_col2:
            if st.button('Réinitialiser pipeline', key=f'reset_pipeline_{ds_clean}'):
                side = 'a' if ds_clean == a_path.name else 'b'
                key = 'pipeline_a' if side == 'a' else 'pipeline_b'
                st.session_state[key] = []
                # persist empty pipeline if processing exists
                _reapply_pipeline_and_persist(side)
                st.success('Pipeline réinitialisé.')

        # Import cleaned CSV: either pick an existing file in `data/` or upload one
        st.markdown('**Importer / Charger une version nettoyée**')
        try:
            # prefer explicit cleaned directory
            cleaned_files = sorted([p.name for p in CLEANED_DIR.glob('*.csv')])
        except Exception:
            cleaned_files = []
        if cleaned_files:
            sel_file = st.selectbox('Fichiers nettoyés trouvés dans `data/`', [''] + cleaned_files, key='sel_clean_file')
            if sel_file:
                if st.button('Charger ce fichier en session', key='load_existing'):
                    path = CLEANED_DIR / sel_file
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
            drop_reason_custom = st.text_input('Raison de suppression (optionnel)', key=f'drop_reason_{ds_clean}', placeholder='Ex: colonnes inutiles, doublons conceptuels...')
            if st.button('Prévisualiser suppression', key=f'preview_drop_{ds_clean}'):
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
            if st.button('Appliquer suppression', key=f'apply_drop_{ds_clean}'):
                if drop_cols:
                    op_payload = {'op':'drop', 'cols': drop_cols, 'reason': 'manual_selection'}
                    if drop_reason_custom:
                        op_payload['reason_text'] = drop_reason_custom
                    if ds_clean == a_path.name:
                        st.session_state['df_a_clean'] = df_clean.drop(columns=drop_cols, errors='ignore')
                        add_op('a', op_payload)
                    else:
                        st.session_state['df_b_clean'] = df_clean.drop(columns=drop_cols, errors='ignore')
                        add_op('b', op_payload)
                    st.success('Suppression appliquée en session.')
                else:
                    st.info('Aucune colonne sélectionnée — rien à appliquer.')
            # Suppression de lignes
            st.markdown('**Supprimer des lignes**')
            row_mode = st.radio('Mode', ['Par index (sélection)', 'Par condition (pandas query)'], key=f'row_del_mode_{ds_clean}')
            if row_mode.startswith('Par index'):
                preview_idx = _make_preview(df_clean, preview_choice).index.tolist()
                sel_rows = st.multiselect('Sélectionner des index à supprimer (aperçu)', preview_idx, key=f'sel_rows_{ds_clean}')
                if st.button('Prévisualiser suppression de lignes', key=f'preview_drop_rows_{ds_clean}'):
                    if not sel_rows:
                        st.info('Aucun index sélectionné pour suppression.')
                    else:
                        preview = _make_preview(df_clean, preview_choice).copy()
                        mask = preview.index.isin(sel_rows)
                        def _hl_row(row):
                            return ['background-color: #ffdddd' if mask.loc[row.name] else '' for _ in row]
                        components.html(preview.style.apply(_hl_row, axis=1).to_html(), height=400, scrolling=True)
                if st.button('Appliquer suppression de lignes', key=f'apply_drop_rows_{ds_clean}'):
                    if not sel_rows:
                        st.info('Aucun index sélectionné — rien à appliquer.')
                    else:
                        if ds_clean == a_path.name:
                            st.session_state['df_a_clean'] = df_clean.drop(index=sel_rows, errors='ignore')
                            add_op('a', {'op': 'drop_rows', 'index': sel_rows})
                        else:
                            st.session_state['df_b_clean'] = df_clean.drop(index=sel_rows, errors='ignore')
                            add_op('b', {'op': 'drop_rows', 'index': sel_rows})
                        st.success(f'{len(sel_rows)} lignes supprimées et ajoutées au pipeline.')
            else:
                cond = st.text_input('Condition pandas (ex: price > 100000 and operation == "sale")', key=f'row_cond_{ds_clean}')
                if st.button('Prévisualiser suppression par condition', key=f'preview_drop_cond_{ds_clean}'):
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
                if st.button('Appliquer suppression par condition', key=f'apply_drop_cond_{ds_clean}'):
                    if not cond:
                        st.info('Aucune condition fournie — rien à appliquer.')
                    else:
                        try:
                            matched_idx = df_clean.query(cond).index
                            if ds_clean == a_path.name:
                                st.session_state['df_a_clean'] = df_clean.drop(index=matched_idx, errors='ignore')
                                add_op('a', {'op': 'drop_rows_cond', 'condition': cond})
                            else:
                                st.session_state['df_b_clean'] = df_clean.drop(index=matched_idx, errors='ignore')
                                add_op('b', {'op': 'drop_rows_cond', 'condition': cond})
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
                if st.button('Prévisualiser colonnes vides (compl.: 0%)', key=f'preview_emptycols0_{ds_clean}'):
                    if not empty_cols:
                        st.info('Aucune colonne complètement vide.')
                    else:
                        preview = _make_preview(df_clean, preview_choice).copy()
                        def _hl_empty(col: pd.Series):
                            if col.name in empty_cols:
                                return ['background-color: #ffdddd' for _ in col]
                            return ['' for _ in col]
                        components.html(preview.style.apply(_hl_empty, axis=0).to_html(), height=400, scrolling=True)
                if st.button('Supprimer colonnes vides (0%)', key=f'drop_emptycols0_{ds_clean}'):
                    if not empty_cols:
                        st.info('Aucune colonne à supprimer.')
                    else:
                        if ds_clean == a_path.name:
                            st.session_state['df_a_clean'] = df_clean.drop(columns=empty_cols, errors='ignore')
                            add_op('a', {'op':'drop', 'cols': empty_cols, 'reason': 'auto_empty_cols_0'})
                        else:
                            st.session_state['df_b_clean'] = df_clean.drop(columns=empty_cols, errors='ignore')
                            add_op('b', {'op':'drop', 'cols': empty_cols, 'reason': 'auto_empty_cols_0'})
                        st.success(f"{len(empty_cols)} colonnes vides supprimées et ajoutées au pipeline.")
            with col_b:
                if st.button('Prévisualiser colonnes complètes (compl.: 100%)', key=f'preview_fullcols100_{ds_clean}'):
                    if not full_cols:
                        st.info('Aucune colonne entièrement complète (100%).')
                    else:
                        preview = _make_preview(df_clean, preview_choice).copy()
                        def _hl_full(col: pd.Series):
                            if col.name in full_cols:
                                return ['background-color: #ffdddd' for _ in col]
                            return ['' for _ in col]
                        components.html(preview.style.apply(_hl_full, axis=0).to_html(), height=400, scrolling=True)
                if st.button('Supprimer colonnes complètes (100%)', key=f'drop_fullcols100_{ds_clean}'):
                    if not full_cols:
                        st.info('Aucune colonne à supprimer.')
                    else:
                        if ds_clean == a_path.name:
                            st.session_state['df_a_clean'] = df_clean.drop(columns=full_cols, errors='ignore')
                            add_op('a', {'op':'drop', 'cols': full_cols, 'reason': 'auto_full_cols_100'})
                        else:
                            st.session_state['df_b_clean'] = df_clean.drop(columns=full_cols, errors='ignore')
                            add_op('b', {'op':'drop', 'cols': full_cols, 'reason': 'auto_full_cols_100'})
                        st.success(f"{len(full_cols)} colonnes complètes supprimées et ajoutées au pipeline.")
            # Supprimer toutes les lignes où une colonne sélectionnée est manquante
            st.markdown('**Supprimer lignes avec valeurs manquantes (colonne)**')
            na_col = st.selectbox('Choisir une colonne (supprimer lignes où NA)', [''] + list(df_clean.columns), key=f'na_col_{ds_clean}')
            if na_col:
                if st.button('Prévisualiser lignes manquantes pour cette colonne', key=f'preview_missingcol_{ds_clean}'):
                    preview = _make_preview(df_clean, preview_choice).copy()
                    mask = preview[na_col].isna()
                    if mask.any():
                        def _hl_na(row):
                            return ['background-color: #ffdddd' if row.name in preview[mask].index else '' for _ in row]
                        components.html(preview.style.apply(_hl_na, axis=1).to_html(), height=400, scrolling=True)
                    else:
                        st.info('Aucune valeur manquante trouvée dans l\'aperçu pour cette colonne.')
                if st.button('Supprimer toutes les lignes où cette colonne est manquante', key=f'drop_missingcol_{ds_clean}'):
                    try:
                        matched_idx = df_clean[df_clean[na_col].isna()].index
                        if matched_idx.empty:
                            st.info('Aucune ligne à supprimer (aucune valeur manquante).')
                        else:
                            if ds_clean == a_path.name:
                                st.session_state['df_a_clean'] = df_clean.drop(index=matched_idx, errors='ignore')
                                add_op('a', {'op': 'drop_rows_where_na', 'col': na_col})
                            else:
                                st.session_state['df_b_clean'] = df_clean.drop(index=matched_idx, errors='ignore')
                                add_op('b', {'op': 'drop_rows_where_na', 'col': na_col})
                            st.success(f'{len(matched_idx)} lignes supprimées et ajoutées au pipeline.')
                    except Exception as e:
                        st.error(f'Erreur lors de la suppression: {e}')
            
            # Scinder colonne catégoriée (ex: "HouseType 0: Estudio" -> col_id + col_desc)
            st.markdown('**Scinder colonne catégoriée**')
            cat_cols_split = df_clean.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
            split_col = st.selectbox('Colonne à scinder (ex: HouseType 0: Estudio)', [''] + cat_cols_split, key=f'split_col_{ds_clean}')
            if split_col:
                split_pattern = st.text_input('Séparateur (ex: :)', value=':', key=f'split_pattern_{ds_clean}')
                split_col_id = st.text_input('Nom colonne ID/Numéro', value=f'{split_col.lower()}_id', key=f'split_col_id_{ds_clean}')
                split_col_desc = st.text_input('Nom colonne Description', value=f'{split_col.lower()}_desc', key=f'split_col_desc_{ds_clean}')
                split_drop_source = st.checkbox('Supprimer la colonne source après scission', value=True, key=f'split_drop_source_{ds_clean}')
                
                if st.button('Prévisualiser scission', key=f'preview_split_{ds_clean}'):
                    preview = _make_preview(df_clean, preview_choice).copy()
                    try:
                        split_result = preview[split_col].astype('string').str.split(split_pattern, n=1, expand=True)
                        if split_result.shape[1] >= 2:
                            preview[split_col_id] = split_result[0].str.extract(r'(\d+)', expand=False).astype('Int64')
                            preview[split_col_desc] = split_result[1].str.strip()
                            cols_to_show = [split_col, split_col_id, split_col_desc]
                            st.dataframe(preview[[c for c in cols_to_show if c in preview.columns]], use_container_width=True)
                        else:
                            st.error(f'Le séparateur "{split_pattern}" n\'a pas été trouvé dans la colonne.')
                    except Exception as e:
                        st.error(f'Erreur prévisualisation scission: {e}')
                
                if st.button('Appliquer scission', key=f'apply_split_{ds_clean}'):
                    try:
                        if ds_clean == a_path.name:
                            df_target = st.session_state['df_a_clean']
                        else:
                            df_target = st.session_state['df_b_clean']
                        
                        split_result = df_target[split_col].astype('string').str.split(split_pattern, n=1, expand=True)
                        if split_result.shape[1] >= 2:
                            df_target[split_col_id] = split_result[0].str.extract(r'(\d+)', expand=False).astype('Int64')
                            df_target[split_col_desc] = split_result[1].str.strip()
                            
                            if split_drop_source:
                                df_target = df_target.drop(columns=[split_col], errors='ignore')
                            
                            if ds_clean == a_path.name:
                                st.session_state['df_a_clean'] = df_target
                            else:
                                st.session_state['df_b_clean'] = df_target
                            
                            add_op('a' if ds_clean == a_path.name else 'b', {
                                'op': 'split_category',
                                'col': split_col,
                                'pattern': split_pattern,
                                'col_id': split_col_id,
                                'col_desc': split_col_desc,
                                'drop_source': split_drop_source
                            })
                            st.success(f'Colonne scindée en {split_col_id} et {split_col_desc}.')
                        else:
                            st.error(f'Le séparateur "{split_pattern}" n\'a pas été trouvé.')
                    except Exception as e:
                        st.error(f'Erreur application scission: {e}')
        with c2:
            st.markdown('**Nettoyage texte simple**')
            text_clean_col = st.selectbox('Colonne (nettoyage texte)', [''] + list(df_clean.columns), key=f'text_clean_col_{ds_clean}')
            text_to_remove = None
            if text_clean_col:
                text_to_remove = st.text_input('Texte à supprimer (exact, sensible à la casse)', key=f'text_to_remove_{ds_clean}', placeholder='Ex: s/n')
                if st.button('Prévisualiser suppression texte', key=f'preview_remove_text_{ds_clean}'):
                    if not text_to_remove:
                        st.warning('Veuillez renseigner le texte à supprimer.')
                    else:
                        preview = _make_preview(df_clean, preview_choice).copy()
                        if text_clean_col in preview.columns:
                            preview[text_clean_col] = preview[text_clean_col].astype('string').str.replace(text_to_remove, '', regex=False).str.strip()
                            st.dataframe(preview[[text_clean_col]], use_container_width=True)
                if st.button('Appliquer suppression texte', key=f'apply_remove_text_{ds_clean}'):
                    if not text_to_remove:
                        st.warning('Veuillez renseigner le texte à supprimer.')
                    else:
                        try:
                            if ds_clean == a_path.name:
                                st.session_state['df_a_clean'][text_clean_col] = st.session_state['df_a_clean'][text_clean_col].astype('string').str.replace(text_to_remove, '', regex=False).str.strip()
                                add_op('a', {'op': 'remove_text', 'col': text_clean_col, 'text': text_to_remove})
                            else:
                                st.session_state['df_b_clean'][text_clean_col] = st.session_state['df_b_clean'][text_clean_col].astype('string').str.replace(text_to_remove, '', regex=False).str.strip()
                                add_op('b', {'op': 'remove_text', 'col': text_clean_col, 'text': text_to_remove})
                            st.success('Suppression du texte appliquée.')
                        except Exception as e:
                            st.error(f'Echec du nettoyage texte: {e}')

            st.markdown('---')
            st.markdown('**Scinder neighborhood_id**')
            nb_cols = df_clean.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
            nb_col = st.selectbox('Colonne à scinder (format Neighborhood X: Name (price €/m2) - District Y: Name)', [''] + nb_cols, key=f'split_nb_col_{ds_clean}')
            if nb_col:
                nb_code_col = st.text_input('Nom colonne code quartier', value='neighborhood_code', key=f'nb_code_col_{ds_clean}')
                nb_name_col = st.text_input('Nom colonne nom quartier', value='neighborhood_name', key=f'nb_name_col_{ds_clean}')
                nb_price_col = st.text_input('Nom colonne prix m2', value='neighborhood_price_sqm', key=f'nb_price_col_{ds_clean}')
                nb_dcode_col = st.text_input('Nom colonne code district', value='district_code', key=f'nb_dcode_col_{ds_clean}')
                nb_dname_col = st.text_input('Nom colonne nom district', value='district_name', key=f'nb_dname_col_{ds_clean}')
                nb_drop_src = st.checkbox('Supprimer la colonne source après scission', value=True, key=f'nb_drop_src_{ds_clean}')

                if st.button('Prévisualiser scission quartier', key=f'preview_split_nb_{ds_clean}'):
                    preview = _make_preview(df_clean, preview_choice).copy()
                    try:
                        pattern = r"Neighborhood\s+(\d+):\s*(.*?)\s*\(([\d\.]+)\s*€/m2\)\s*-\s*District\s+(\d+):\s*(.*)"
                        extracted = preview[nb_col].astype('string').str.extract(pattern)
                        if extracted.shape[1] == 5:
                            preview[nb_code_col] = extracted[0].astype('Int64')
                            preview[nb_name_col] = extracted[1].astype('string').str.strip()
                            preview[nb_price_col] = pd.to_numeric(extracted[2], errors='coerce')
                            preview[nb_dcode_col] = extracted[3].astype('Int64')
                            preview[nb_dname_col] = extracted[4].astype('string').str.strip()
                            cols_show = [c for c in [nb_col, nb_code_col, nb_name_col, nb_price_col, nb_dcode_col, nb_dname_col] if c in preview.columns]
                            st.dataframe(preview[cols_show], use_container_width=True)
                        else:
                            st.warning('Le format attendu n\'a pas été reconnu dans l\'aperçu.')
                    except Exception as e:
                        st.error(f'Erreur prévisualisation scission quartier: {e}')

                if st.button('Appliquer scission quartier', key=f'apply_split_nb_{ds_clean}'):
                    try:
                        if ds_clean == a_path.name:
                            df_target = st.session_state['df_a_clean']
                        else:
                            df_target = st.session_state['df_b_clean']

                        pattern = r"Neighborhood\s+(\d+):\s*(.*?)\s*\(([\d\.]+)\s*€/m2\)\s*-\s*District\s+(\d+):\s*(.*)"
                        extracted = df_target[nb_col].astype('string').str.extract(pattern)
                        if extracted.shape[1] == 5:
                            df_target[nb_code_col] = extracted[0].astype('Int64')
                            df_target[nb_name_col] = extracted[1].astype('string').str.strip()
                            df_target[nb_price_col] = pd.to_numeric(extracted[2], errors='coerce')
                            df_target[nb_dcode_col] = extracted[3].astype('Int64')
                            df_target[nb_dname_col] = extracted[4].astype('string').str.strip()
                            if nb_drop_src:
                                df_target = df_target.drop(columns=[nb_col], errors='ignore')
                            if ds_clean == a_path.name:
                                st.session_state['df_a_clean'] = df_target
                                add_op('a', {
                                    'op': 'split_neighborhood',
                                    'col': nb_col,
                                    'code_col': nb_code_col,
                                    'name_col': nb_name_col,
                                    'price_col': nb_price_col,
                                    'dist_code_col': nb_dcode_col,
                                    'dist_name_col': nb_dname_col,
                                    'drop_source': nb_drop_src
                                })
                            else:
                                st.session_state['df_b_clean'] = df_target
                                add_op('b', {
                                    'op': 'split_neighborhood',
                                    'col': nb_col,
                                    'code_col': nb_code_col,
                                    'name_col': nb_name_col,
                                    'price_col': nb_price_col,
                                    'dist_code_col': nb_dcode_col,
                                    'dist_name_col': nb_dname_col,
                                    'drop_source': nb_drop_src
                                })
                            st.success('Scission quartier appliquée et ajoutée au pipeline.')
                        else:
                            st.warning('Le format attendu n\'a pas été reconnu. Aucune modification appliquée.')
                    except Exception as e:
                        st.error(f'Erreur application scission quartier: {e}')

            st.markdown('---')
            st.markdown('**Imputation — built_year par groupes (district + neighborhood)**')
            if 'built_year' in df_clean.columns:
                if st.button('Prévisualiser imputation built_year', key=f'preview_impute_built_year_{ds_clean}'):
                    preview = _make_preview(df_clean, preview_choice).copy()
                    mask_na = preview['built_year'].isna()
                    if not mask_na.any():
                        st.info("Aucune ligne à imputer dans l'aperçu.")
                    else:
                        try:
                            # Grouper par district_name, neighborhood_name, neighborhood_code
                            groupby_cols = []
                            for col in ['district_name', 'neighborhood_name', 'neighborhood_code']:
                                if col in df_clean.columns:
                                    groupby_cols.append(col)
                            
                            if groupby_cols:
                                grp = df_clean.groupby(groupby_cols)['built_year'].agg(lambda s: s.mode(dropna=True).iloc[0] if not s.mode(dropna=True).empty else None)
                                vals = preview.loc[mask_na, groupby_cols].apply(lambda row: grp.get(tuple(row), None), axis=1)
                                to_show = preview.loc[mask_na, ['built_year'] + groupby_cols].copy()
                                to_show['année_imputée'] = vals
                                to_show['built_year_imputed_flag'] = 1
                                st.dataframe(to_show, use_container_width=True)
                            else:
                                st.warning('Colonnes de regroupement (district_name, neighborhood_name, neighborhood_code) non trouvées.')
                        except Exception as e:
                            st.error(f'Erreur prévisualisation built_year: {e}')
                
                if st.button('Appliquer imputation built_year', key=f'apply_impute_built_year_{ds_clean}'):
                    try:
                        if ds_clean == a_path.name:
                            df_target = st.session_state['df_a_clean']
                        else:
                            df_target = st.session_state['df_b_clean']
                        
                        groupby_cols = []
                        for col in ['district_name', 'neighborhood_name', 'neighborhood_code']:
                            if col in df_target.columns:
                                groupby_cols.append(col)
                        
                        if groupby_cols:
                            mask_na = df_target['built_year'].isna()
                            if mask_na.any():
                                grp = df_target.groupby(groupby_cols)['built_year'].agg(lambda s: s.mode(dropna=True).iloc[0] if not s.mode(dropna=True).empty else None)
                                vals = df_target.loc[mask_na, groupby_cols].apply(lambda row: grp.get(tuple(row), None), axis=1)
                                df_target.loc[mask_na, 'built_year'] = vals
                                flag_col = 'built_year_imputed_flag'
                                df_target[flag_col] = df_target[flag_col] if flag_col in df_target.columns else 0
                                df_target.loc[mask_na, flag_col] = 1
                                cols = list(df_target.columns)
                                if flag_col in cols:
                                    cols.remove(flag_col)
                                    try:
                                        idx = cols.index('built_year')
                                        cols.insert(idx + 1, flag_col)
                                        df_target = df_target[cols]
                                    except ValueError:
                                        pass
                                
                                if ds_clean == a_path.name:
                                    st.session_state['df_a_clean'] = df_target
                                    add_op('a', {'op': 'impute_mode_multi_group', 'col': 'built_year', 'groupby_cols': groupby_cols, 'flag_col': flag_col})
                                else:
                                    st.session_state['df_b_clean'] = df_target
                                    add_op('b', {'op': 'impute_mode_multi_group', 'col': 'built_year', 'groupby_cols': groupby_cols, 'flag_col': flag_col})
                                st.success(f'Imputation built_year appliquée: {mask_na.sum()} lignes imputées avec flag.')
                            else:
                                st.info('Aucune valeur manquante à imputer.')
                        else:
                            st.warning('Colonnes de regroupement (district_name, neighborhood_name, neighborhood_code) non trouvées.')
                    except Exception as e:
                        st.error(f'Erreur application imputation built_year: {e}')
            else:
                st.warning('Colonne built_year non trouvée.')

            st.markdown('**Imputation — Colonnes numériques**')
            num_cols = df_clean.select_dtypes(include='number').columns.tolist()
            impute_col = st.selectbox('Imputer (numérique) — choisir colonne', [''] + num_cols)
            impute_strategy = st.selectbox('Stratégie', ['médiane', 'moyenne', 'constante', 'calcul', 'mode (global)', 'mode par groupe'])
            const_val = None
            if impute_strategy == 'constante':
                const_val = st.text_input('Valeur constante (ex: 0)')
            calc_num = None
            calc_den = None
            calc_round = 'float'
            mode_group_col = None
            if impute_strategy == 'calcul':
                # Choose numerator/denominator columns for calculation
                calc_num = st.selectbox('Numérateur (colonne)', [''] + num_cols, index=(num_cols.index('buy_price') + 1) if 'buy_price' in num_cols else 0, key=f'impute_calc_num_{ds_clean}')
                calc_den = st.selectbox('Dénominateur (colonne)', [''] + num_cols, index=(num_cols.index('buy_price_by_area') + 1) if 'buy_price_by_area' in num_cols else 0, key=f'impute_calc_den_{ds_clean}')
                calc_round = st.selectbox('Type de sortie', ['float', 'int'], index=0, key=f'impute_calc_round_{ds_clean}')
            if impute_strategy == 'mode (global)':
                st.info('Le mode global remplira les NA avec la modalité la plus fréquente de la colonne.')
            if impute_strategy == 'mode par groupe':
                # proposer les colonnes catégorielles/numériques entières comme regroupement
                cat_or_int = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()
                cat_or_int += [c for c in df_clean.select_dtypes(include='integer').columns if c not in cat_or_int]
                mode_group_col = st.selectbox('Colonne de regroupement pour le mode', [''] + cat_or_int, key=f'impute_mode_group_{ds_clean}')
            
            st.markdown('---')
            st.markdown('**Transformation — Fusion de colonnes booléennes**')
            bool_cols = [c for c in df_clean.columns if df_clean[c].dtype == 'bool' or str(df_clean[c].dtype) == 'boolean']
            if len(bool_cols) >= 2:
                st.markdown('Fusionner deux colonnes booléennes exclusives en une colonne catégorielle.')
                merge_bool_central = st.selectbox('Colonne "central" (True → central)', [''] + bool_cols, key=f'merge_bool_central_{ds_clean}')
                merge_bool_individual = st.selectbox('Colonne "individual" (True → individual)', [''] + bool_cols, key=f'merge_bool_individual_{ds_clean}')
                merge_bool_target = st.text_input('Nom de la nouvelle colonne catégorielle', value='heating_type', key=f'merge_bool_target_{ds_clean}')
                merge_bool_drop = st.checkbox('Supprimer les colonnes sources', value=True, key=f'merge_bool_drop_{ds_clean}')
                
                if merge_bool_central and merge_bool_individual and merge_bool_target and merge_bool_central != merge_bool_individual:
                    if st.button('Prévisualiser fusion booléenne', key=f'preview_merge_bool_{ds_clean}'):
                        preview = _make_preview(df_clean, preview_choice).copy()
                        try:
                            preview[merge_bool_target] = 'unknown'
                            preview.loc[preview[merge_bool_central] == True, merge_bool_target] = 'central'
                            preview.loc[preview[merge_bool_individual] == True, merge_bool_target] = 'individual'
                            mask_both_na = preview[merge_bool_central].isna() & preview[merge_bool_individual].isna()
                            preview.loc[mask_both_na, merge_bool_target] = pd.NA
                            
                            cols_to_show = [merge_bool_central, merge_bool_individual, merge_bool_target]
                            st.dataframe(preview[cols_to_show].value_counts(dropna=False).reset_index(), use_container_width=True)
                            st.info(f"Valeurs NA dans {merge_bool_target}: {preview[merge_bool_target].isna().sum()}")
                        except Exception as e:
                            st.error(f'Erreur prévisualisation: {e}')
                    
                    if st.button('Appliquer fusion booléenne', key=f'apply_merge_bool_{ds_clean}'):
                        try:
                            if ds_clean == a_path.name:
                                df_target = st.session_state['df_a_clean']
                            else:
                                df_target = st.session_state['df_b_clean']
                            
                            # Créer la nouvelle colonne
                            df_target[merge_bool_target] = 'unknown'
                            df_target.loc[df_target[merge_bool_central] == True, merge_bool_target] = 'central'
                            df_target.loc[df_target[merge_bool_individual] == True, merge_bool_target] = 'individual'
                            mask_both_na = df_target[merge_bool_central].isna() & df_target[merge_bool_individual].isna()
                            df_target.loc[mask_both_na, merge_bool_target] = pd.NA
                            
                            # Supprimer les colonnes sources si demandé
                            if merge_bool_drop:
                                df_target = df_target.drop(columns=[merge_bool_central, merge_bool_individual])
                            
                            if ds_clean == a_path.name:
                                st.session_state['df_a_clean'] = df_target
                                add_op('a', {'op': 'merge_bool_to_category', 'col_central': merge_bool_central, 'col_individual': merge_bool_individual, 'col_target': merge_bool_target, 'drop_sources': merge_bool_drop})
                            else:
                                st.session_state['df_b_clean'] = df_target
                                add_op('b', {'op': 'merge_bool_to_category', 'col_central': merge_bool_central, 'col_individual': merge_bool_individual, 'col_target': merge_bool_target, 'drop_sources': merge_bool_drop})
                            
                            action = "+ colonnes sources supprimées" if merge_bool_drop else ""
                            st.success(f'Fusion appliquée: {merge_bool_target} créée {action}')
                        except Exception as e:
                            st.error(f'Erreur application fusion: {e}')
            else:
                st.info('Au moins 2 colonnes booléennes requises.')

            st.markdown('---')
            st.markdown('**Imputation — Colonnes booléennes (logique inverse)**')
            bool_cols = [c for c in df_clean.columns if df_clean[c].dtype == 'bool' or str(df_clean[c].dtype) == 'boolean']
            if len(bool_cols) >= 2:
                impute_bool_target = st.selectbox('Colonne cible à imputer (bool)', [''] + bool_cols, key=f'impute_bool_target_{ds_clean}')
                impute_bool_source = st.selectbox('Colonne source (l\'inverse sera utilisé)', [''] + bool_cols, key=f'impute_bool_source_{ds_clean}')
                if impute_bool_target and impute_bool_source and impute_bool_target != impute_bool_source:
                    if st.button('Prévisualiser imputation booléenne inverse', key=f'preview_impute_bool_{ds_clean}'):
                        preview = _make_preview(df_clean, preview_choice).copy()
                        try:
                            inverted = ~preview[impute_bool_source].astype(bool)
                            to_show = preview[[impute_bool_target, impute_bool_source]].copy()
                            to_show['valeur_transformée'] = inverted
                            to_show[f"{impute_bool_target}_imputed_flag"] = 1
                            st.info(f"Transformation: {impute_bool_target} = NOT {impute_bool_source} ({len(to_show)} lignes)")
                            st.dataframe(to_show, use_container_width=True)
                        except Exception as e:
                            st.error(f'Erreur prévisualisation: {e}')
                    
                    if st.button('Appliquer imputation booléenne inverse', key=f'apply_impute_bool_{ds_clean}'):
                        try:
                            flag_col_bool = f"{impute_bool_target}_imputed_flag"
                            if ds_clean == a_path.name:
                                df_target = st.session_state['df_a_clean']
                            else:
                                df_target = st.session_state['df_b_clean']
                            
                            # Transformer TOUTES les valeurs avec l'inverse booléen
                            inverted = ~df_target[impute_bool_source].astype(bool)
                            df_target[impute_bool_target] = inverted
                            df_target[flag_col_bool] = df_target[flag_col_bool] if flag_col_bool in df_target.columns else 0
                            df_target[flag_col_bool] = 1  # Tous les éléments ont été transformés
                            cols = list(df_target.columns)
                            if flag_col_bool in cols:
                                cols.remove(flag_col_bool)
                                try:
                                    idx = cols.index(impute_bool_target)
                                    cols.insert(idx + 1, flag_col_bool)
                                    df_target = df_target[cols]
                                except ValueError:
                                    pass
                            
                            if ds_clean == a_path.name:
                                st.session_state['df_a_clean'] = df_target
                                add_op('a', {'op': 'impute_inverse_bool', 'col_target': impute_bool_target, 'col_source': impute_bool_source, 'flag_col': flag_col_bool})
                            else:
                                st.session_state['df_b_clean'] = df_target
                                add_op('b', {'op': 'impute_inverse_bool', 'col_target': impute_bool_target, 'col_source': impute_bool_source, 'flag_col': flag_col_bool})
                            st.success(f'Transformation booléenne appliquée: toutes les {len(df_target)} lignes transformées avec flag.')
                        except Exception as e:
                            st.error(f'Erreur application transformation booléenne: {e}')
            else:
                st.info('Au moins 2 colonnes booléennes requises pour cette imputation.')

            st.markdown('---')
            st.markdown('**Imputation — Colonnes catégorielles**')
            cat_cols = df_clean.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
            impute_cat_col = st.selectbox('Imputer (catégorie) — choisir colonne', [''] + cat_cols, key=f'impute_cat_col_{ds_clean}')
            impute_cat_val = None
            if impute_cat_col:
                impute_cat_val = st.text_input('Valeur de remplissage (catégorie)', key=f'impute_cat_val_{ds_clean}', placeholder='Ex: Unknown, N/A, etc.')
            if st.button('Prévisualiser imputation', key=f'preview_impute_{ds_clean}') and impute_col:
                preview_full = _make_preview(df_clean, preview_choice).copy()
                # Build filtered view showing only rows that would be imputed
                if impute_strategy == 'calcul':
                    if not calc_num or not calc_den:
                        st.error('Veuillez sélectionner un numérateur et un dénominateur.')
                    else:
                        try:
                            num_ser = pd.to_numeric(preview_full[calc_num], errors='coerce')
                            den_ser = pd.to_numeric(preview_full[calc_den], errors='coerce')
                            calc = num_ser / den_ser
                            if calc_round == 'int':
                                calc = calc.round().astype('Int64')
                            else:
                                calc = calc.astype(float)
                            mask_na = preview_full[impute_col].isna() & calc.notna()
                            if not mask_na.any():
                                st.info("Aucune ligne à imputer dans l'aperçu.")
                            else:
                                to_show = preview_full.loc[mask_na, [impute_col, calc_num, calc_den]].copy()
                                to_show['valeur_calculée'] = calc[mask_na]
                                to_show[f"{impute_col}_imputed_flag"] = 1
                                st.dataframe(to_show, use_container_width=True)
                        except Exception as e:
                            st.error(f'Erreur calcul imputation: {e}')
                else:
                    skip_generic = False
                    if impute_strategy == 'médiane':
                        val = df_clean[impute_col].median()
                    elif impute_strategy == 'moyenne':
                        val = df_clean[impute_col].mean()
                    elif impute_strategy.startswith('mode'):
                        if impute_strategy == 'mode (global)':
                            mode_series = df_clean[impute_col].mode(dropna=True)
                            val = mode_series.iloc[0] if not mode_series.empty else None
                            if val is None:
                                st.info("Impossible de calculer le mode global.")
                                skip_generic = True
                            else:
                                mask_na0 = preview_full[impute_col].isna()
                                if not mask_na0.any():
                                    st.info("Aucune ligne à imputer dans l'aperçu.")
                                    skip_generic = True
                                else:
                                    preview_full = preview_full.copy()
                                    preview_full[impute_col] = preview_full[impute_col].fillna(val)
                                    st.dataframe(preview_full.loc[mask_na0, [impute_col]], use_container_width=True)
                                    skip_generic = True
                        else:
                            if not mode_group_col:
                                st.error('Sélectionnez une colonne de regroupement pour le mode.')
                                skip_generic = True
                            else:
                                na_mask = preview_full[impute_col].isna()
                                if not na_mask.any():
                                    st.info("Aucune ligne à imputer dans l'aperçu.")
                                else:
                                    grp = df_clean.groupby(mode_group_col)[impute_col].agg(lambda s: s.mode(dropna=True).iloc[0] if not s.mode(dropna=True).empty else None)
                                    vals = preview_full.loc[na_mask, mode_group_col].map(grp)
                                    to_show = preview_full.loc[na_mask, [impute_col, mode_group_col]].copy()
                                    to_show['valeur_imputee'] = vals
                                    to_show[f"{impute_col}_imputed_flag"] = 1
                                    st.dataframe(to_show, use_container_width=True)
                                skip_generic = True
                        # skip_generic prevents falling through to generic fill
                        val = None
                    else:
                        try:
                            val = float(const_val)
                        except Exception:
                            val = const_val
                    if not skip_generic:
                        mask_na0 = preview_full[impute_col].isna()
                        if not mask_na0.any():
                            st.info("Aucune ligne à imputer dans l'aperçu.")
                        else:
                            preview_full = preview_full.copy()
                            preview_full[impute_col] = preview_full[impute_col].fillna(val)
                            to_show = preview_full.loc[mask_na0, [impute_col]].copy()
                            to_show[f"{impute_col}_imputed_flag"] = 1
                            st.dataframe(to_show, use_container_width=True)
            if st.button('Appliquer imputation', key=f'apply_impute_{ds_clean}') and impute_col:
                flag_col = f"{impute_col}_imputed_flag"
                if impute_strategy == 'calcul':
                    if not calc_num or not calc_den:
                        st.error('Veuillez sélectionner un numérateur et un dénominateur.')
                    else:
                        try:
                            if ds_clean == a_path.name:
                                df_target = st.session_state['df_a_clean']
                                num_ser = pd.to_numeric(df_target[calc_num], errors='coerce')
                                den_ser = pd.to_numeric(df_target[calc_den], errors='coerce')
                                calc = num_ser / den_ser
                                if calc_round == 'int':
                                    calc = calc.round().astype('Int64')
                                else:
                                    calc = calc.astype(float)
                                mask_na = df_target[impute_col].isna() & calc.notna()
                                df_target.loc[mask_na, impute_col] = calc[mask_na]
                                df_target[flag_col] = df_target[flag_col] if flag_col in df_target.columns else 0
                                df_target.loc[mask_na, flag_col] = 1
                                # ensure column order places flag after target
                                cols = list(df_target.columns)
                                if flag_col in cols:
                                    cols.remove(flag_col)
                                    try:
                                        idx = cols.index(impute_col)
                                        cols.insert(idx + 1, flag_col)
                                        df_target = df_target[cols]
                                    except ValueError:
                                        pass
                                st.session_state['df_a_clean'] = df_target
                                add_op('a', {'op':'impute_calc', 'col': impute_col, 'numerator': calc_num, 'denominator': calc_den, 'round': calc_round, 'flag_col': flag_col})
                            else:
                                df_target = st.session_state['df_b_clean']
                                num_ser = pd.to_numeric(df_target[calc_num], errors='coerce')
                                den_ser = pd.to_numeric(df_target[calc_den], errors='coerce')
                                calc = num_ser / den_ser
                                if calc_round == 'int':
                                    calc = calc.round().astype('Int64')
                                else:
                                    calc = calc.astype(float)
                                mask_na = df_target[impute_col].isna() & calc.notna()
                                df_target.loc[mask_na, impute_col] = calc[mask_na]
                                df_target[flag_col] = df_target[flag_col] if flag_col in df_target.columns else 0
                                df_target.loc[mask_na, flag_col] = 1
                                cols = list(df_target.columns)
                                if flag_col in cols:
                                    cols.remove(flag_col)
                                    try:
                                        idx = cols.index(impute_col)
                                        cols.insert(idx + 1, flag_col)
                                        df_target = df_target[cols]
                                    except ValueError:
                                        pass
                                st.session_state['df_b_clean'] = df_target
                                add_op('b', {'op':'impute_calc', 'col': impute_col, 'numerator': calc_num, 'denominator': calc_den, 'round': calc_round, 'flag_col': flag_col})
                            st.success('Imputation calculée appliquée et ajoutée au pipeline.')
                        except Exception as e:
                            st.error(f'Erreur lors de l\'imputation calculée: {e}')
                else:
                    if impute_strategy == 'médiane':
                        val = df_clean[impute_col].median()
                        op = {'op':'impute', 'col': impute_col, 'val': val, 'strategy': 'median', 'flag_col': flag_col}
                        if ds_clean == a_path.name:
                            df_target = st.session_state['df_a_clean']
                            mask_na = df_target[impute_col].isna()
                            df_target[impute_col] = df_target[impute_col].fillna(val)
                            df_target[flag_col] = df_target[flag_col] if flag_col in df_target.columns else 0
                            df_target.loc[mask_na, flag_col] = 1
                            cols = list(df_target.columns)
                            if flag_col in cols:
                                cols.remove(flag_col)
                                try:
                                    idx = cols.index(impute_col)
                                    cols.insert(idx + 1, flag_col)
                                    df_target = df_target[cols]
                                except ValueError:
                                    pass
                            st.session_state['df_a_clean'] = df_target
                            add_op('a', op)
                        else:
                            df_target = st.session_state['df_b_clean']
                            mask_na = df_target[impute_col].isna()
                            df_target[impute_col] = df_target[impute_col].fillna(val)
                            df_target[flag_col] = df_target[flag_col] if flag_col in df_target.columns else 0
                            df_target.loc[mask_na, flag_col] = 1
                            cols = list(df_target.columns)
                            if flag_col in cols:
                                cols.remove(flag_col)
                                try:
                                    idx = cols.index(impute_col)
                                    cols.insert(idx + 1, flag_col)
                                    df_target = df_target[cols]
                                except ValueError:
                                    pass
                            st.session_state['df_b_clean'] = df_target
                            add_op('b', op)
                        st.success('Imputation appliquée en session (médiane).')
                    elif impute_strategy == 'moyenne':
                        val = df_clean[impute_col].mean()
                        op = {'op':'impute', 'col': impute_col, 'val': val, 'strategy': 'mean', 'flag_col': flag_col}
                        if ds_clean == a_path.name:
                            df_target = st.session_state['df_a_clean']
                            mask_na = df_target[impute_col].isna()
                            df_target[impute_col] = df_target[impute_col].fillna(val)
                            df_target[flag_col] = df_target[flag_col] if flag_col in df_target.columns else 0
                            df_target.loc[mask_na, flag_col] = 1
                            cols = list(df_target.columns)
                            if flag_col in cols:
                                cols.remove(flag_col)
                                try:
                                    idx = cols.index(impute_col)
                                    cols.insert(idx + 1, flag_col)
                                    df_target = df_target[cols]
                                except ValueError:
                                    pass
                            st.session_state['df_a_clean'] = df_target
                            add_op('a', op)
                        else:
                            df_target = st.session_state['df_b_clean']
                            mask_na = df_target[impute_col].isna()
                            df_target[impute_col] = df_target[impute_col].fillna(val)
                            df_target[flag_col] = df_target[flag_col] if flag_col in df_target.columns else 0
                            df_target.loc[mask_na, flag_col] = 1
                            cols = list(df_target.columns)
                            if flag_col in cols:
                                cols.remove(flag_col)
                                try:
                                    idx = cols.index(impute_col)
                                    cols.insert(idx + 1, flag_col)
                                    df_target = df_target[cols]
                                except ValueError:
                                    pass
                            st.session_state['df_b_clean'] = df_target
                            add_op('b', op)
                        st.success('Imputation appliquée en session (moyenne).')
                    elif impute_strategy == 'mode (global)':
                        mode_series = df_clean[impute_col].mode(dropna=True)
                        if mode_series.empty:
                            st.error("Impossible de calculer le mode global.")
                        else:
                            val = mode_series.iloc[0]
                            op = {'op':'fillna_mode', 'cols': [impute_col], 'flag_col': flag_col}
                            if ds_clean == a_path.name:
                                df_target = st.session_state['df_a_clean']
                                mask_na = df_target[impute_col].isna()
                                df_target[impute_col] = df_target[impute_col].fillna(val)
                                df_target[flag_col] = df_target[flag_col] if flag_col in df_target.columns else 0
                                df_target.loc[mask_na, flag_col] = 1
                                cols = list(df_target.columns)
                                if flag_col in cols:
                                    cols.remove(flag_col)
                                    try:
                                        idx = cols.index(impute_col)
                                        cols.insert(idx + 1, flag_col)
                                        df_target = df_target[cols]
                                    except ValueError:
                                        pass
                                st.session_state['df_a_clean'] = df_target
                                add_op('a', op)
                            else:
                                df_target = st.session_state['df_b_clean']
                                mask_na = df_target[impute_col].isna()
                                df_target[impute_col] = df_target[impute_col].fillna(val)
                                df_target[flag_col] = df_target[flag_col] if flag_col in df_target.columns else 0
                                df_target.loc[mask_na, flag_col] = 1
                                cols = list(df_target.columns)
                                if flag_col in cols:
                                    cols.remove(flag_col)
                                    try:
                                        idx = cols.index(impute_col)
                                        cols.insert(idx + 1, flag_col)
                                        df_target = df_target[cols]
                                    except ValueError:
                                        pass
                                st.session_state['df_b_clean'] = df_target
                                add_op('b', op)
                            st.success('Imputation appliquée en session (mode global).')
                    elif impute_strategy == 'mode par groupe':
                        if not mode_group_col:
                            st.error('Sélectionnez une colonne de regroupement pour le mode.')
                        else:
                            try:
                                if ds_clean == a_path.name:
                                    df_target = st.session_state['df_a_clean']
                                    grp = df_target.groupby(mode_group_col)[impute_col].agg(lambda s: s.mode(dropna=True).iloc[0] if not s.mode(dropna=True).empty else None)
                                    mask_na = df_target[impute_col].isna()
                                    vals = df_target.loc[mask_na, mode_group_col].map(grp)
                                    df_target.loc[mask_na, impute_col] = vals
                                    df_target[flag_col] = df_target[flag_col] if flag_col in df_target.columns else 0
                                    df_target.loc[mask_na, flag_col] = 1
                                    cols = list(df_target.columns)
                                    if flag_col in cols:
                                        cols.remove(flag_col)
                                        try:
                                            idx = cols.index(impute_col)
                                            cols.insert(idx + 1, flag_col)
                                            df_target = df_target[cols]
                                        except ValueError:
                                            pass
                                    st.session_state['df_a_clean'] = df_target
                                    add_op('a', {'op':'fillna_mode', 'cols': [impute_col], 'groupby': mode_group_col, 'flag_col': flag_col})
                                else:
                                    df_target = st.session_state['df_b_clean']
                                    grp = df_target.groupby(mode_group_col)[impute_col].agg(lambda s: s.mode(dropna=True).iloc[0] if not s.mode(dropna=True).empty else None)
                                    mask_na = df_target[impute_col].isna()
                                    vals = df_target.loc[mask_na, mode_group_col].map(grp)
                                    df_target.loc[mask_na, impute_col] = vals
                                    df_target[flag_col] = df_target[flag_col] if flag_col in df_target.columns else 0
                                    df_target.loc[mask_na, flag_col] = 1
                                    cols = list(df_target.columns)
                                    if flag_col in cols:
                                        cols.remove(flag_col)
                                        try:
                                            idx = cols.index(impute_col)
                                            cols.insert(idx + 1, flag_col)
                                            df_target = df_target[cols]
                                        except ValueError:
                                            pass
                                    st.session_state['df_b_clean'] = df_target
                                    add_op('b', {'op':'fillna_mode', 'cols': [impute_col], 'groupby': mode_group_col, 'flag_col': flag_col})
                                st.success('Imputation appliquée en session (mode par groupe).')
                            except Exception as e:
                                st.error(f'Erreur lors de l\'imputation par mode: {e}')
                    else:
                        try:
                            val = float(const_val)
                        except Exception:
                            val = const_val
                        op = {'op':'impute', 'col': impute_col, 'val': val, 'strategy': 'const', 'flag_col': flag_col}
                        if ds_clean == a_path.name:
                            df_target = st.session_state['df_a_clean']
                            mask_na = df_target[impute_col].isna()
                            df_target[impute_col] = df_target[impute_col].fillna(val)
                            df_target[flag_col] = df_target[flag_col] if flag_col in df_target.columns else 0
                            df_target.loc[mask_na, flag_col] = 1
                            cols = list(df_target.columns)
                            if flag_col in cols:
                                cols.remove(flag_col)
                                try:
                                    idx = cols.index(impute_col)
                                    cols.insert(idx + 1, flag_col)
                                    df_target = df_target[cols]
                                except ValueError:
                                    pass
                            st.session_state['df_a_clean'] = df_target
                            add_op('a', op)
                        else:
                            df_target = st.session_state['df_b_clean']
                            mask_na = df_target[impute_col].isna()
                            df_target[impute_col] = df_target[impute_col].fillna(val)
                            df_target[flag_col] = df_target[flag_col] if flag_col in df_target.columns else 0
                            df_target.loc[mask_na, flag_col] = 1
                            cols = list(df_target.columns)
                            if flag_col in cols:
                                cols.remove(flag_col)
                                try:
                                    idx = cols.index(impute_col)
                                    cols.insert(idx + 1, flag_col)
                                    df_target = df_target[cols]
                                except ValueError:
                                    pass
                            st.session_state['df_b_clean'] = df_target
                            add_op('b', op)
                        st.success('Imputation appliquée en session (constante).')

            # Imputation pour colonnes catégorielles
            if impute_cat_col:
                if st.button('Prévisualiser imputation catégorie', key=f'preview_impute_cat_{ds_clean}'):
                    preview_full = _make_preview(df_clean, preview_choice).copy()
                    mask_na = preview_full[impute_cat_col].isna()
                    if not mask_na.any():
                        st.info("Aucune ligne à imputer dans l'aperçu.")
                    else:
                        to_show = preview_full.loc[mask_na, [impute_cat_col]].copy()
                        to_show['valeur_imputée'] = impute_cat_val
                        to_show[f"{impute_cat_col}_imputed_flag"] = 1
                        st.dataframe(to_show, use_container_width=True)
                if st.button('Appliquer imputation catégorie', key=f'apply_impute_cat_{ds_clean}'):
                    if not impute_cat_val:
                        st.warning('Veuillez entrer une valeur de remplissage.')
                    else:
                        flag_col_cat = f"{impute_cat_col}_imputed_flag"
                        if ds_clean == a_path.name:
                            df_target = st.session_state['df_a_clean']
                            mask_na = df_target[impute_cat_col].isna()
                            df_target[impute_cat_col] = df_target[impute_cat_col].fillna(impute_cat_val)
                            df_target[flag_col_cat] = df_target[flag_col_cat] if flag_col_cat in df_target.columns else 0
                            df_target.loc[mask_na, flag_col_cat] = 1
                            cols = list(df_target.columns)
                            if flag_col_cat in cols:
                                cols.remove(flag_col_cat)
                                try:
                                    idx = cols.index(impute_cat_col)
                                    cols.insert(idx + 1, flag_col_cat)
                                    df_target = df_target[cols]
                                except ValueError:
                                    pass
                            st.session_state['df_a_clean'] = df_target
                            add_op('a', {'op':'impute', 'col': impute_cat_col, 'val': impute_cat_val, 'strategy': 'const', 'flag_col': flag_col_cat})
                        else:
                            df_target = st.session_state['df_b_clean']
                            mask_na = df_target[impute_cat_col].isna()
                            df_target[impute_cat_col] = df_target[impute_cat_col].fillna(impute_cat_val)
                            df_target[flag_col_cat] = df_target[flag_col_cat] if flag_col_cat in df_target.columns else 0
                            df_target.loc[mask_na, flag_col_cat] = 1
                            cols = list(df_target.columns)
                            if flag_col_cat in cols:
                                cols.remove(flag_col_cat)
                                try:
                                    idx = cols.index(impute_cat_col)
                                    cols.insert(idx + 1, flag_col_cat)
                                    df_target = df_target[cols]
                                except ValueError:
                                    pass
                            st.session_state['df_b_clean'] = df_target
                            add_op('b', {'op':'impute', 'col': impute_cat_col, 'val': impute_cat_val, 'strategy': 'const', 'flag_col': flag_col_cat})
                        st.success('Imputation catégorie appliquée en session.')

            # Imputation par extraction de texte
            st.markdown('---')
            st.markdown('**Imputation — Extraction de texte**')
            text_cols = df_clean.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
            impute_extract_source = st.selectbox('Colonne source (extraire du texte)', [''] + text_cols, key=f'impute_extract_source_{ds_clean}')
            impute_extract_target = st.selectbox('Colonne cible (à remplir)', [''] + text_cols, key=f'impute_extract_target_{ds_clean}')
            impute_extract_pattern = None
            if impute_extract_source and impute_extract_target:
                impute_extract_pattern = st.text_input('Pattern/Séparateur (ex: "en venta en ")', key=f'impute_extract_pattern_{ds_clean}', placeholder='Texte après lequel extraire')
                
                if st.button('Prévisualiser extraction', key=f'preview_extract_{ds_clean}'):
                    if not impute_extract_pattern:
                        st.error('Veuillez entrer un pattern.')
                    else:
                        preview = _make_preview(df_clean, preview_choice).copy()
                        mask_na = preview[impute_extract_target].isna()
                        if not mask_na.any():
                            st.info("Aucune ligne à imputer dans l'aperçu.")
                        else:
                            try:
                                extracted = preview[impute_extract_source].astype('string').str.split(impute_extract_pattern, n=1, expand=True)
                                if extracted.shape[1] >= 2:
                                    extracted_text = extracted[1].str.strip()
                                    to_show = preview.loc[mask_na, [impute_extract_target, impute_extract_source]].copy()
                                    to_show['extraction'] = extracted_text[mask_na]
                                    to_show[f"{impute_extract_target}_imputed_flag"] = 1
                                    st.dataframe(to_show, use_container_width=True)
                                else:
                                    st.warning(f'Pattern "{impute_extract_pattern}" non trouvé dans la colonne source.')
                            except Exception as e:
                                st.error(f'Erreur prévisualisation: {e}')
                
                if st.button('Appliquer extraction', key=f'apply_extract_{ds_clean}'):
                    if not impute_extract_pattern:
                        st.error('Veuillez entrer un pattern.')
                    else:
                        try:
                            flag_col_extract = f"{impute_extract_target}_imputed_flag"
                            if ds_clean == a_path.name:
                                df_target = st.session_state['df_a_clean']
                            else:
                                df_target = st.session_state['df_b_clean']
                            
                            mask_na = df_target[impute_extract_target].isna()
                            extracted = df_target[impute_extract_source].astype('string').str.split(impute_extract_pattern, n=1, expand=True)
                            if extracted.shape[1] >= 2:
                                extracted_text = extracted[1].str.strip()
                                df_target.loc[mask_na, impute_extract_target] = extracted_text[mask_na]
                                df_target[flag_col_extract] = df_target[flag_col_extract] if flag_col_extract in df_target.columns else 0
                                df_target.loc[mask_na, flag_col_extract] = 1
                                cols = list(df_target.columns)
                                if flag_col_extract in cols:
                                    cols.remove(flag_col_extract)
                                    try:
                                        idx = cols.index(impute_extract_target)
                                        cols.insert(idx + 1, flag_col_extract)
                                        df_target = df_target[cols]
                                    except ValueError:
                                        pass
                                
                                if ds_clean == a_path.name:
                                    st.session_state['df_a_clean'] = df_target
                                    add_op('a', {'op':'impute_extract_text', 'col_source': impute_extract_source, 'col_target': impute_extract_target, 'extract_pattern': impute_extract_pattern, 'flag_col': flag_col_extract})
                                else:
                                    st.session_state['df_b_clean'] = df_target
                                    add_op('b', {'op':'impute_extract_text', 'col_source': impute_extract_source, 'col_target': impute_extract_target, 'extract_pattern': impute_extract_pattern, 'flag_col': flag_col_extract})
                                st.success(f'Extraction appliquée: {mask_na.sum()} lignes imputées.')
                            else:
                                st.warning(f'Pattern "{impute_extract_pattern}" non trouvé.')
                        except Exception as e:
                            st.error(f'Erreur application extraction: {e}')

            # Typage & Normalisation
            st.markdown('**Typage & Normalisation**')
            # Typage (caster les colonnes)
            cast_cols = st.multiselect('Colonnes à caster', list(df_clean.columns), key=f'cast_cols_{ds_clean}')
            cast_dtype = st.selectbox('Type cible', ['int', 'float', 'datetime', 'bool', 'category', 'string'], key=f'cast_dtype_{ds_clean}')
            if st.button('Prévisualiser typage', key=f'preview_cast_{ds_clean}') and cast_cols:
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
            if st.button('Appliquer typage', key=f'apply_cast_{ds_clean}') and cast_cols:
                def _dtype_short_cast(ser: pd.Series) -> str:
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

                prev_types: dict[str, str] = {}
                for col in cast_cols:
                    try:
                        if col in df_clean.columns:
                            prev_types[col] = _dtype_short_cast(df_clean[col])
                    except Exception:
                        prev_types[col] = None
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
                    add_op('a', {'op': 'cast', 'cols': cast_cols, 'dtype': cast_dtype, 'dtype_prev': prev_types})
                else:
                    add_op('b', {'op': 'cast', 'cols': cast_cols, 'dtype': cast_dtype, 'dtype_prev': prev_types})
                st.success('Typage appliqué et ajouté au pipeline.')

            # Normalisation (scaling)
            st.markdown('**Normalisation (colonnes numériques)**')
            scale_cols = st.multiselect('Colonnes numériques à normaliser', num_cols, key=f'scale_cols_{ds_clean}')
            scale_method = st.selectbox('Méthode de normalisation', ['Standard (z-score)', 'Min-Max (0-1)'], key=f'scale_method_{ds_clean}')
            method_key = 'standard' if scale_method.startswith('Standard') else 'minmax'
            if st.button('Prévisualiser normalisation', key=f'preview_scale_{ds_clean}') and scale_cols:
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
            if st.button('Appliquer normalisation', key=f'apply_scale_{ds_clean}') and scale_cols:
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
                    add_op('a', {'op': 'scale', 'cols': scale_cols, 'method': method_key, 'params': params})
                else:
                    add_op('b', {'op': 'scale', 'cols': scale_cols, 'method': method_key, 'params': params})
                st.success('Normalisation appliquée et ajoutée au pipeline.')

            # Détection d'incohérences (validation croisée)
            st.markdown('**Validation de cohérence — Superficie (sq_mt_built vs buy_price/buy_price_by_area)**')
            tol_pct = st.slider('Tolérance relative (%)', min_value=1, max_value=80, value=20, step=1, key=f'coherence_tol_{ds_clean}')
            st.markdown('_Vérifie que la superficie déclarée correspond à celle calculée par prix total / prix au m² (tolérance par défaut 20%)_')
            
            coherence_expr = f'abs(sq_mt_built - (buy_price / buy_price_by_area)) / sq_mt_built * 100 > {tol_pct}'
            
            coherence_reason = st.text_input(
                'Description optionnelle',
                placeholder='Ex: divergence superficie calculée vs déclarée',
                key=f'coherence_reason_{ds_clean}'
            )
            
            col_coh_a, col_coh_b = st.columns(2)
            with col_coh_a:
                if st.button('Prévisualiser incohérences', key=f'preview_coherence_{ds_clean}'):
                    try:
                        # Check if required columns exist
                        required_cols = ['sq_mt_built', 'buy_price', 'buy_price_by_area']
                        missing = [c for c in required_cols if c not in df_clean.columns]
                        if missing:
                            st.error(f'Colonnes manquantes: {", ".join(missing)}')
                        else:
                            # Evaluate expression to find incoherent rows
                            incoherent = df_clean.query(coherence_expr)
                            if incoherent.empty:
                                st.success(f'✓ Aucune incohérence détectée ({len(df_clean)} lignes vérifiées).')
                            else:
                                st.warning(f'⚠ {len(incoherent)} ligne(s) incohérente(s) détectée(s) sur {len(df_clean)}.')
                                # Affiche colonnes pertinentes + calcul
                                preview_cols = ['sq_mt_built', 'buy_price', 'buy_price_by_area']
                                preview_cols = [c for c in preview_cols if c in incoherent.columns]
                                preview = incoherent[preview_cols].copy()
                                preview['surface_calculated'] = (incoherent['buy_price'] / incoherent['buy_price_by_area']).round(2)
                                preview['divergence_%'] = (abs(incoherent['sq_mt_built'] - preview['surface_calculated']) / incoherent['sq_mt_built'] * 100).round(2)
                                st.dataframe(_make_preview(preview, preview_choice), use_container_width=True)
                    except Exception as e:
                        st.error(f'Erreur: {e}')
            
            with col_coh_b:
                if st.button('Supprimer lignes incohérentes', key=f'apply_coherence_{ds_clean}'):
                    try:
                        required_cols = ['sq_mt_built', 'buy_price', 'buy_price_by_area']
                        missing = [c for c in required_cols if c not in df_clean.columns]
                        if missing:
                            st.error(f'Colonnes manquantes: {", ".join(missing)}')
                        else:
                            incoherent_idx = df_clean.query(coherence_expr).index
                            if incoherent_idx.empty:
                                st.info('Aucune ligne incohérente à supprimer.')
                            else:
                                op_payload = {
                                    'op': 'drop_rows_cond',
                                    'condition': coherence_expr,
                                    'coherence_check': True,
                                    'coherence_type': 'surface_validation',
                                    'tolerance_pct': tol_pct
                                }
                                if coherence_reason:
                                    op_payload['reason_text'] = coherence_reason
                                
                                if ds_clean == a_path.name:
                                    st.session_state['df_a_clean'] = df_clean.drop(index=incoherent_idx, errors='ignore')
                                    add_op('a', op_payload)
                                else:
                                    st.session_state['df_b_clean'] = df_clean.drop(index=incoherent_idx, errors='ignore')
                                    add_op('b', op_payload)
                                st.success(f'{len(incoherent_idx)} ligne(s) incohérente(s) supprimée(s).')
                    except Exception as e:
                        st.error(f'Erreur lors de la suppression: {e}')

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
            if st.button('Prévisualiser doublons', key=f'preview_dups_{ds_clean}'):
                # Filtrer uniquement les lignes dupliquées (sur le périmètre choisi)
                dup_df = df_clean.loc[dup_mask_full]
                if dup_df.empty:
                    st.info('Aucun doublon détecté selon les colonnes sélectionnées.')
                else:
                    # Réutilise la logique d'aperçu pour limiter le volume affiché
                    preview = _make_preview(dup_df, preview_choice).copy()
                    st.dataframe(preview, use_container_width=True)
            if st.button('Compter doublons', key=f'count_dups_{ds_clean}'):
                total_dup_groups = dup_mask_full.sum()
                st.info(f'Lignes impliquées dans des doublons (keep=False): {int(total_dup_groups)} / {len(df_clean)}')
        with col_dup_b:
            if st.button('Supprimer doublons (garder premier)', key=f'drop_dups_first_{ds_clean}'):
                if dup_subset:
                    cols = dup_subset
                else:
                    cols = None
                if ds_clean == a_path.name:
                    st.session_state['df_a_clean'] = df_clean.drop_duplicates(subset=cols, keep='first')
                    add_op('a', {'op': 'drop_duplicates', 'subset': cols or [], 'keep': 'first'})
                else:
                    st.session_state['df_b_clean'] = df_clean.drop_duplicates(subset=cols, keep='first')
                    add_op('b', {'op': 'drop_duplicates', 'subset': cols or [], 'keep': 'first'})
                st.success('Doublons supprimés (garder premier) et ajoutés au pipeline.')
            if st.button('Supprimer doublons (garder dernier)', key=f'drop_dups_last_{ds_clean}'):
                if dup_subset:
                    cols = dup_subset
                else:
                    cols = None
                if ds_clean == a_path.name:
                    st.session_state['df_a_clean'] = df_clean.drop_duplicates(subset=cols, keep='last')
                    add_op('a', {'op': 'drop_duplicates', 'subset': cols or [], 'keep': 'last'})
                else:
                    st.session_state['df_b_clean'] = df_clean.drop_duplicates(subset=cols, keep='last')
                    add_op('b', {'op': 'drop_duplicates', 'subset': cols or [], 'keep': 'last'})
                st.success('Doublons supprimés (garder dernier) et ajoutés au pipeline.')
            if st.button('Supprimer toutes lignes dupliquées', key=f'drop_dups_all_{ds_clean}'):
                if dup_subset:
                    cols = dup_subset
                else:
                    cols = None
                # keep=False removes all rows that are duplicated
                if ds_clean == a_path.name:
                    st.session_state['df_a_clean'] = df_clean.loc[~dup_mask_full].copy()
                    add_op('a', {'op': 'drop_duplicates', 'subset': cols or [], 'keep': False})
                else:
                    st.session_state['df_b_clean'] = df_clean.loc[~dup_mask_full].copy()
                    add_op('b', {'op': 'drop_duplicates', 'subset': cols or [], 'keep': False})
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
            if st.button('Prévisualiser modification', key=f'preview_edit_{ds_clean}'):
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
            if st.button('Appliquer modification', key=f'apply_edit_{ds_clean}'):
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
                            add_op('a', {'op': 'set_value', 'index': chosen_idx, 'col': edit_col, 'val': casted_val})
                        else:
                            st.session_state['df_b_clean'].at[chosen_idx, edit_col] = casted_val
                            add_op('b', {'op': 'set_value', 'index': chosen_idx, 'col': edit_col, 'val': casted_val})
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
            if st.button('Annuler dernière opération', key=f'undo_preview_{ds_clean}'):
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
            if st.button('Réinitialiser tout', key=f'reset_all_{ds_clean}'):
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
            # Enregistrer côté serveur dans data/cleaned (CSV + pipeline JSON avec métadonnées)
            if st.button('Enregistrer version nettoyée sur le serveur (data/cleaned)', key=f'save_clean_server_{ds_clean}'):
                try:
                    cleaned_dir = CLEANED_DIR
                    cleaned_dir.mkdir(parents=True, exist_ok=True)
                    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                    csv_path = cleaned_dir / f"{a_path.stem}.cleaned.{ts}.csv"
                    # atomic CSV write
                    _atomic_write_csv(csv_path, st.session_state['df_a_clean'])
                    # build pipeline metadata JSON
                    pipeline_ops = st.session_state.get('pipeline_a', []) or []
                    origin_raw = st.session_state.get('current_raw_a') or a_path.name
                    orig_rows = int((df_a.shape[0] if isinstance(df_a, pd.DataFrame) else 0))
                    cleaned_rows = int(st.session_state['df_a_clean'].shape[0])
                    meta = {
                        'origin_raw': origin_raw,
                        'cleaned_csv': csv_path.name,
                        'created_at': datetime.now().isoformat(timespec='seconds'),
                        'ops_count': len([op for op in pipeline_ops if op.get('op')]),
                        'rows_removed': max(0, orig_rows - cleaned_rows),
                        'pipeline': pipeline_ops,
                    }
                    json_path = cleaned_dir / f"{a_path.stem}.cleaned.{ts}.pipeline.json"
                    _atomic_write_json(json_path, meta)
                    st.success(f'Fichiers enregistrés: {csv_path.name} et {json_path.name}')
                except Exception as e:
                    st.error(f"Échec sauvegarde serveur: {e}")
        else:
            csv = st.session_state['df_b_clean'].to_csv(index=False)
            st.download_button('Télécharger version nettoyée (CSV)', csv, file_name=f'{b_path.stem}.cleaned.csv')
            # Enregistrer côté serveur dans data/cleaned (CSV + pipeline JSON avec métadonnées)
            if st.button('Enregistrer version nettoyée sur le serveur (data/cleaned)', key=f'save_clean_server_upload_{ds_clean}'):
                try:
                    cleaned_dir = CLEANED_DIR
                    cleaned_dir.mkdir(parents=True, exist_ok=True)
                    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                    csv_path = cleaned_dir / f"{b_path.stem}.cleaned.{ts}.csv"
                    _atomic_write_csv(csv_path, st.session_state['df_b_clean'])
                    pipeline_ops = st.session_state.get('pipeline_b', []) or []
                    origin_raw = st.session_state.get('current_raw_b') or b_path.name
                    orig_rows = int((df_b.shape[0] if isinstance(df_b, pd.DataFrame) else 0))
                    cleaned_rows = int(st.session_state['df_b_clean'].shape[0])
                    meta = {
                        'origin_raw': origin_raw,
                        'cleaned_csv': csv_path.name,
                        'created_at': datetime.now().isoformat(timespec='seconds'),
                        'ops_count': len([op for op in pipeline_ops if op.get('op')]),
                        'rows_removed': max(0, orig_rows - cleaned_rows),
                        'pipeline': pipeline_ops,
                    }
                    json_path = cleaned_dir / f"{b_path.stem}.cleaned.{ts}.pipeline.json"
                    _atomic_write_json(json_path, meta)
                    st.success(f'Fichiers enregistrés: {csv_path.name} et {json_path.name}')
                except Exception as e:
                    st.error(f"Échec sauvegarde serveur: {e}")

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
            # compute per-op row removals (prefer cached rows_removed if present)
            row_breakdown = []
            df_sim = orig_df.copy()
            for op in pipeline_ops:
                if not op.get('op'):
                    continue
                removed_here = op.get('rows_removed')
                if removed_here is None:
                    try:
                        df_next = apply_pipeline(df_sim, [op])
                        removed_here = int(df_sim.shape[0] - df_next.shape[0])
                        df_sim = df_next
                    except Exception:
                        removed_here = None
                else:
                    try:
                        df_sim = apply_pipeline(df_sim, [op])
                    except Exception:
                        pass
                row_breakdown.append((op, removed_here))

            lines.append("## Détail des suppressions de lignes")
            any_rows = any((v or 0) > 0 for (_, v) in row_breakdown)
            if not any_rows:
                lines.append("Aucune suppression de lignes enregistrée.")
            else:
                for idx, (op, cnt) in enumerate(row_breakdown, 1):
                    if not op.get('op'):
                        continue
                    typ = op.get('op')
                    if typ == 'drop_rows':
                        lines.append(f"{idx}. Lignes supprimées (index explicites): {cnt if cnt is not None else 'N/A'} — index: {op.get('index', [])}")
                    elif typ == 'drop_rows_cond':
                        lines.append(f"{idx}. Lignes supprimées (condition): {cnt if cnt is not None else 'N/A'} — condition: {op.get('condition', '')}")
                    elif typ == 'drop_rows_where_na':
                        lines.append(f"{idx}. Lignes supprimées (valeurs manquantes): {cnt if cnt is not None else 'N/A'} — colonne: {op.get('col', '')}")
                    elif typ == 'drop_duplicates':
                        subset = op.get('subset', []) or 'toutes les colonnes'
                        keep = op.get('keep', 'first')
                        lines.append(f"{idx}. Lignes supprimées (doublons): {cnt if cnt is not None else 'N/A'} — colonnes: {subset} — action: {keep}")
                    else:
                        # no row effect expected
                        continue

            lines.append("")
            lines.append("## Détail des opérations")
            if not pipeline_ops:
                lines.append("Aucune opération enregistrée.")
            else:
                for i, op in enumerate(pipeline_ops, 1):
                    desc = _describe_op(op)
                    if op.get('rows_removed') is not None:
                        desc = f"{desc} — lignes supprimées: {op.get('rows_removed')}"
                    lines.append(f"{i}. {desc}")
            lines.append("")
            lines.append("## Pipeline (JSON)")
            lines.append("```json")
            lines.append(json.dumps(pipeline_ops, indent=2, default=str))
            lines.append("```")
            return "\n".join(lines)

        orig_df = df_a if ds_clean == a_path.name else df_b
        cleaned_df = st.session_state['df_a_clean'] if ds_clean == a_path.name else st.session_state['df_b_clean']
        pipeline_ops = pipeline or []
        pipeline_ops = _annotate_pipeline_with_row_diffs(orig_df, pipeline_ops)
        report_md = _generate_report(ds_clean, pipeline_ops, orig_df, cleaned_df)

        if st.button('Générer compte rendu (aperçu)', key=f'gen_report_{ds_clean}'):
            with st.expander('Aperçu du compte rendu', expanded=True):
                st.markdown(report_md)

        # Téléchargements: markdown et pipeline JSON
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        st.download_button('Télécharger compte rendu (Markdown)', report_md, file_name=f'compte_rendu_{ds_clean}_{timestamp}.md', mime='text/markdown')
        raw_for_meta = st.session_state.get('current_raw_a') if ds_clean == a_path.name else st.session_state.get('current_raw_b')
        proc_for_meta = proc_name if use_processing and proc_name else ''
        meta_extra = _make_meta_metrics(orig_df, cleaned_df, pipeline_ops)
        pipeline_to_dl = _pipeline_with_meta(raw_for_meta or ds_clean, proc_for_meta or '', pipeline_ops, meta_extra=meta_extra)
        st.download_button('Télécharger pipeline (JSON)', json.dumps(pipeline_to_dl, indent=2, default=str), file_name=f'pipeline_{ds_clean}_{timestamp}.json', mime='application/json')

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

        # Scores & recommandation pour les versions nettoyées
        try:
            info_clean_a = summarize(cleaned_a)
            info_clean_b = summarize(cleaned_b)
            score_ca = 0.0
            score_cb = 0.0
            tgt_ca = detect_target(cleaned_a) is not None
            tgt_cb = detect_target(cleaned_b) is not None
            if tgt_ca:
                score_ca += w_target
            if tgt_cb:
                score_cb += w_target
            rows_max_clean = max(info_clean_a['rows'], info_clean_b['rows']) if max(info_clean_a['rows'], info_clean_b['rows']) > 0 else 1
            score_ca += w_rows * (info_clean_a['rows'] / rows_max_clean)
            score_cb += w_rows * (info_clean_b['rows'] / rows_max_clean)
            miss_max_pct_c = max(info_clean_a['missing_perc'], info_clean_b['missing_perc'], 1.0)
            out_max_pct_c = max(info_clean_a.get('outlier_perc', 0.0), info_clean_b.get('outlier_perc', 0.0), 1.0)
            score_ca += w_missing * (1 - info_clean_a['missing_perc'] / miss_max_pct_c)
            score_cb += w_missing * (1 - info_clean_b['missing_perc'] / miss_max_pct_c)
            score_ca += w_outlier * (1 - info_clean_a.get('outlier_perc', 0.0) / out_max_pct_c)
            score_cb += w_outlier * (1 - info_clean_b.get('outlier_perc', 0.0) / out_max_pct_c)

            st.markdown('---')
            st.subheader('Recommandation & Scores — versions nettoyées')
            winner_clean = a_path.name if score_ca >= score_cb else b_path.name
            st.markdown(f"### ✅ Recommandation (nettoyé): **{winner_clean}**")
            max_score_c = max(score_ca, score_cb, 1e-6)
            nca = score_ca / max_score_c
            ncb = score_cb / max_score_c
            colorCA = _interp_color(nca, low_col, high_col)
            colorCB = _interp_color(ncb, low_col, high_col)
            colCA, colCB = st.columns(2)
            with colCA:
                st.markdown(horizontal_bar(nca, colorCA, f"{a_path.name} — {score_ca:.2f}"), unsafe_allow_html=True)
            with colCB:
                st.markdown(horizontal_bar(ncb, colorCB, f"{b_path.name} — {score_cb:.2f}"), unsafe_allow_html=True)
        except Exception:
            st.info('Impossible de calculer les scores pour les versions nettoyées (données manquantes).')


if __name__ == '__main__':
    main()
