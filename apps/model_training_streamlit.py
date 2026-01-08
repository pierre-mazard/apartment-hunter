from __future__ import annotations

import json
import sys
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.ml.pipelines import (
    infer_feature_sets,
    build_preprocessor,
    get_model,
    build_pipeline,
    compute_metrics,
    cross_validate_pipeline,
    get_param_distributions,
    save_model_and_meta,
    split_data,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CLEANED_DIR = DATA_DIR / "cleaned"
MODELS_DIR = ROOT / "models"
DOCS_DIR = ROOT / "docs"
RESULTS_DIR = ROOT / "results"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def guess_target_candidates(df: pd.DataFrame) -> list[str]:
    """Retourne des candidats de cible ordonnés (prix/vente/location d'abord)."""
    if df.empty or df.shape[1] == 0:
        return []

    tokens = ["price", "prix", "precio", "sale", "buy", "purchase", "rent"]
    price_like = [c for c in df.columns if any(t in c.lower() for t in tokens)]

    # Fallback: colonnes numériques contenant "price" si rien n'a matché
    if not price_like:
        for c in df.select_dtypes(include="number").columns:
            if "price" in c.lower() or "precio" in c.lower():
                price_like.append(c)

    # Si toujours rien, propose simplement la dernière colonne
    if not price_like:
        return [df.columns[-1]]

    # Score = moyenne des corrélations absolues avec les autres numériques + bonus de priorité sémantique
    numeric = df.select_dtypes(include="number")
    corr = None
    if numeric.shape[1] >= 2:
        try:
            corr = numeric.corr().abs()
        except Exception:
            corr = None

    priority = ["sale", "buy", "purchase", "price", "precio", "rent"]
    scores: dict[str, float] = {}
    for col in price_like:
        score = 0.0
        if corr is not None and col in corr.columns:
            col_corr = corr[col].drop(labels=[col], errors="ignore")
            score += float(col_corr.mean(skipna=True)) if not col_corr.empty else 0.0
        for idx, token in enumerate(priority):
            if token in col.lower():
                score += (len(priority) - idx) * 0.01
                break
        scores[col] = score

    ordered = sorted(price_like, key=lambda c: scores.get(c, 0.0), reverse=True)
    # Ajoute la dernière colonne si elle n'est pas déjà dedans pour garder une option neutre
    if df.columns[-1] not in ordered:
        ordered.append(df.columns[-1])
    # Supprime les doublons en préservant l'ordre
    seen = set()
    uniq_ordered = []
    for c in ordered:
        if c not in seen:
            uniq_ordered.append(c)
            seen.add(c)
    return uniq_ordered


def guess_target_column(df: pd.DataFrame) -> str:
    candidates = guess_target_candidates(df)
    return candidates[0] if candidates else ""

st.set_page_config(page_title="Entraînement & comparaison de modèles", layout="wide")
st.title("Entraînement & comparaison de modèles")
st.markdown("Sélectionnez un CSV nettoyé, choisissez la cible, la tâche et comparez plusieurs modèles avec validation croisée.")

# Aide rapide
st.markdown("""
**Guide d'utilisation rapide**
- Chargez un CSV de `data/cleaned/` (nettoyé depuis l'autre app).
- Choisissez la cible : la tâche (régression vs classification) est détectée automatiquement.
- Laissez les réglages par défaut pour un run rapide : test 20%, validation croisée sur 5 folds, tous les modèles proposés.
- Activez la recherche d'hyperparamètres uniquement si le dataset est assez grand (>2k lignes) ou si vous cherchez le meilleur score.
""")

st.markdown("---")
st.markdown("""

**Séparation des données:**
- **Train/Test split** : divise les données en ensemble d'entraînement (par défaut 80%) et test (20%)
  - Permet une évaluation "honnête" sur des données jamais vues
  - `test_size`: ajustez pour votre besoin (ex: 0.2 = 20% test)

**Validation Croisée (K-Fold):**
- Divise l'ensemble **train** en **K** folds (groupes)
- Entraîne K modèles, chacun en laissant un fold de côté pour validation
- Moyenne les scores → plus robuste qu'une seule éval
- Par défaut: **K=5** folds (test sur 5 groupes différents)
- Formule: chaque modèle voit 4/5 des données en train, 1/5 en validation
- **Important** : Même nombre de folds utilisé pour tous les modèles (avec ou sans recherche d'hyperparamètres)

**Recherche d'hyperparamètres (RandomizedSearchCV):**
- Teste automatiquement **plusieurs configurations** d'hyperparamètres pour RF et GB
- Utilise la **même validation croisée K-Fold** que sans recherche (comparaison équitable)
- Pour chaque configuration testée : calcule RMSE, MAE, R² (ou accuracy/F1) sur les K folds
- Sélectionne la **meilleure configuration** selon R² (régression) ou F1 (classification)
- **Pas de double CV** : les métriques finales proviennent directement de la recherche
- Linear/Logistic : pas d'hyperparamètres à optimiser → CV standard
- Exemple : 8 itérations = teste 8 combinaisons différentes d'hyperparamètres

**Repeated Hold-Out (option):**
- Au lieu d'une seule séparation train/test, relancer **n** fois avec des splits différents
- Chaque itération: nouveau split train/test aléatoire → n évaluations test
- Moyenne les n résultats test pour une variance plus stable
- Par défaut: désactivé (n=1, split unique)
""")

st.markdown("---")

with st.expander("📊 Modèles disponibles et explications", expanded=False):
    st.markdown("""
### 📈 Régression (prédiction de valeurs continues)

#### Linear Regression
- **Principe**: modèle baseline qui cherche une relation linéaire entre features et cible
- **Équation**: `y = w₀ + w₁×x₁ + w₂×x₂ + ... + wₙ×xₙ` (hyperplan)
- **Avantages**: ⚡ Très rapide | 🔍 Facile à interpréter | 📊 Bon baseline
- **Inconvénients**: ❌ Suppose linéarité | ❌ Sensible aux outliers

#### Random Forest (Régression)
- **Principe**: ensemble de **N arbres décisionnels** qui votent ensemble
- **Fonctionnement**: Chaque arbre entraîné sur bootstrap → **prédictions numériques indépendantes** → **moyenne finale**
- **Avantages**: 🌳 Robuste | 🛡️ Résiste outliers | 🎯 Bon compromis
- **Inconvénients**: 🐢 Plus lent | 🌫️ Moins interprétable

#### Gradient Boosting (Régression)
- **Principe**: arbres **séquentiels** où chaque nouvel arbre **corrige les erreurs numériques** (résidus) du précédent
- **Fonctionnement**: Arbre #1 → calcul résidus (vraie_valeur - prédiction) → Arbre #2 prédit ces résidus → répéter
- **Avantages**: 🏆 Très performant | 🎯 Ajustement fin | 📈 Patterns complexes
- **Inconvénients**: ⏱️ Plus lent | ⚠️ Risque surapprentissage

---

### 🎯 Classification (prédiction de catégories/classes)

#### Logistic Regression
- **Principe**: complètement différent de Linear ! Linéaire + fonction sigmoid pour probabilités 0-1
- **Équation**: `P(classe 1) = sigmoid(w₀ + w₁×x₁ + ... + wₙ×xₙ)` où sigmoid(z) = 1/(1+e⁻ᶻ)
- **Avantages**: ⚡ Très rapide | 🔍 Interprétable | 📊 Probabilités calibrées
- **Inconvénients**: ❌ Suppose linéarité | ❌ Moins bon sur données complexes

#### Random Forest (Classification)
- **Principe**: **Même architecture que régression**, mais fonctionnement différent
- **Fonctionnement**: Chaque arbre prédit une **classe** → **classe la plus votée** = prédiction finale (pas moyenne!)
- **Avantages**: 🌳 Gère non-linéarité | 🛡️ Robuste | ⚖️ Moins sensible déséquilibre
- **Inconvénients**: 🐢 Plus lent | 🌫️ Peu interprétable

#### Gradient Boosting (Classification)
- **Principe**: **Même architecture que régression**, mais corrige les **erreurs de classification** au lieu de résidus numériques
- **Fonctionnement**: Arbre #1 → identifie points mal classifiés → Arbre #2 apprend sur ces points problématiques → répéter
- **Avantages**: 🏆 Très performant | 🎯 Ajustement fin | ⚖️ Bon si bien configuré
- **Inconvénients**: ⏱️ Lent | ⚠️ Surapprentissage | 🌫️ Peu interprétable

---

### ⚙️ Différences clés entre régression et classification pour Random Forest / Gradient Boosting

| Aspect | Régression | Classification |
|--------|-----------|-----------------|
| **Prédiction arbre RF** | Valeur numérique | Classe (catégorie) |
| **Combinaison arbres RF** | **Moyenne** des valeurs | **Vote majoritaire** (mode) |
| **Erreur corrigée GB** | Résidu numérique (y_vrai - y_pred) | Points mal classifiés / probabilités erronées |
| **Sortie finale** | Nombre continu | Classe + probabilités |

**💡 Remarque**: Linear et Logistic sont des modèles **fondamentalement différents** (une équation pour régression, une autre avec sigmoid pour classification). Mais RF et GB ont la **même structure algorithmique**, adaptée à chaque tâche.

---

### 🎯 Quel modèle choisir ?

| Priorité | Modèle recommandé |
|----------|------------------|
| **Performance max** | Gradient Boosting |
| **Équilibre vitesse/perf** | Random Forest |
| **Interprétabilité** | Logistic/Linear |
| **Peu de données (<500)** | Linear/Logistic |
| **Beaucoup de données (>10k)** | GB ou RF |
| **Benchmark rapide** | Linear/Logistic |

**💡 Astuce**: Commencez par Linear/Logistic baseline, puis testez RF et GB pour voir le gain.
    """)

# Choix du fichier nettoyé
try:
    cleaned_files = sorted([p.name for p in CLEANED_DIR.glob("*.csv")])
except Exception:
    cleaned_files = []

if not cleaned_files:
    st.warning("Aucun fichier dans data/cleaned. Exportez d'abord une version nettoyée.")
    st.stop()

sel_file = st.selectbox("Jeu de données nettoyé", cleaned_files, key="sel_cleaned")
path = CLEANED_DIR / sel_file

df = pd.read_csv(path)
st.write(f"Aperçu ({len(df)} lignes, {df.shape[1]} colonnes)")
st.dataframe(df.head(200), width='stretch')

# Choix de la cible et détection automatique de la tâche
cols = df.columns.tolist()
if not cols:
    st.error("Dataset vide")
    st.stop()

target_candidates = guess_target_candidates(df)
initial_target = target_candidates[0] if target_candidates else cols[-1]
options = list(dict.fromkeys(target_candidates + cols))  # candidats en tête, puis toutes les colonnes
sel_target = st.selectbox(
    "Colonne cible (détection automatique modifiable)",
    options,
    index=options.index(initial_target),
    key="target",
)
if target_candidates:
    st.caption(f"Cibles proposées (ordre de préférence): {', '.join(target_candidates[:5])}")
    st.markdown(
        "<small>Ordre basé sur les noms contenant price/prix/sale/buy/rent et la corrélation moyenne avec les autres variables numériques.</small>",
        unsafe_allow_html=True,
    )

# Heuristique: si cible numérique avec beaucoup de valeurs distinctes -> régression, sinon classification
n_unique_target = df[sel_target].nunique(dropna=True)
is_numeric_target = pd.api.types.is_numeric_dtype(df[sel_target])
task_auto = "regression" if (is_numeric_target and n_unique_target > 10) else "classification"
st.markdown(f"Type de tâche détecté automatiquement: **{task_auto}** (cible: {n_unique_target} valeurs distinctes, numérique: {is_numeric_target})")
st.caption("Conseil: laissez l'autodétection sauf cas particulier (cible numérique à classes discrètes -> classification forcée).")

# Option d'override (rarement nécessaire)
override = st.checkbox("Forcer le type de tâche (optionnel)", value=False, key="override_task")
if override:
    task = st.radio("Tâche", ["regression", "classification"], index=0 if task_auto == "regression" else 1, key="task")
else:
    task = task_auto

models_reg = ["linear", "random_forest", "gradient_boosting"]
models_clf = ["logistic_regression", "random_forest", "gradient_boosting"]
model_options = models_reg if task == "regression" else models_clf
sel_models: List[str] = st.multiselect("Modèles à comparer", model_options, default=model_options, key="models")
st.caption("Régression: linear = baseline rapide ; random_forest = robuste ; gradient_boosting = bon compromis biais/variance. Classification: logistic = baseline, RF = robuste, GB = performant si tuning.")

col_split, col_cv, col_search = st.columns(3)
with col_split:
    test_size = st.slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05, key="test_size")
    st.caption("0.2 par défaut. Si très peu de données (<2k lignes), rester entre 0.2 et 0.25.")
with col_cv:
    cv_folds = st.number_input("Folds de validation croisée", min_value=2, max_value=10, value=5, step=1, key="cv_folds")
    st.caption("5 par défaut. Réduire à 3 si dataset petit pour gagner du temps.")
with col_search:
    do_search = st.checkbox("Recherche d'hyperparamètres (RandomizedSearchCV)", value=False, key="do_search")
    search_iters = st.number_input("Iterations", min_value=1, max_value=30, value=8, step=1, key="search_iters") if do_search else 0
    st.caption("Activez seulement si vous avez du volume (>2k lignes) ou besoin d'affiner. 8-10 itérations est un bon début.")

st.markdown("---")
st.subheader("Options avancées")
col_rep, col_space = st.columns(2)
with col_rep:
    do_repeated = st.checkbox("Évaluation répétée (Repeated Hold-Out)", value=False, key="do_repeated")
    n_repeats = st.number_input("Nombre de répétitions", min_value=2, max_value=10, value=3, step=1, key="n_repeats") if do_repeated else 1
    st.caption("Multiples évaluations test indépendantes pour moyenne ± std. Augmente le temps de calcul.")
with col_space:
    if do_repeated:
        st.info(f"Vous aurez {n_repeats} split(s) test indépendant(s).")

st.markdown("---")
st.subheader("Sauvegarde des modèles")
save_mode = st.radio("Quel(s) modèle(s) sauvegarder ?", ["Seulement le meilleur", "Tous les modèles"], index=0)
save_all_models = (save_mode == "Tous les modèles")
st.caption("Si 'Tous', chaque modèle entraîné sera sauvegardé dans models/ (utile pour comparaison ultérieure).")

run_btn = st.button("Lancer le pipeline", type="primary")

# Afficher le dernier run en mémoire (si la page se rerend après le clic, on garde une trace)
if "last_run" in st.session_state:
    lr = st.session_state["last_run"]
    st.subheader("Dernier run (mémoire)")
    try:
        _mem_df = pd.DataFrame(lr.get("rows", []))
        if not _mem_df.empty:
            _mem_df = _mem_df.astype(str)
        st.dataframe(_mem_df, width='stretch')
    except Exception:
        st.write("(Impossible d'afficher le tableau en mémoire)")
    meta = lr.get("meta", {})
    if meta:
        st.caption(f"Dataset: {meta.get('data')} — cible: {meta.get('target')} — run_id: {meta.get('run_id')}")
    saved = []
    if lr.get("csv"):
        saved.append(lr["csv"])
    if lr.get("json"):
        saved.append(lr["json"])
    if saved:
        st.caption(f"Fichiers sauvegardés: {', '.join(saved)}")
    st.markdown("---")

if run_btn:
    if not sel_models:
        st.error("Sélectionnez au moins un modèle")
        st.stop()

    num_cols, cat_cols = infer_feature_sets(df, sel_target)
    preprocessor = build_preprocessor(num_cols, cat_cols)

    # Splits: une ou plusieurs répétitions
    n_test_repeats = int(n_repeats) if do_repeated else 1
    splits_list = []
    for rep in range(n_test_repeats):
        rs = 42 + rep if do_repeated else 42
        split = split_data(df, sel_target, test_size=test_size, random_state=rs, task=task)
        splits_list.append(split)

    # Pour la première répétition, garder les indices pour la sauvegarde meta
    train_indices = splits_list[0].X_train.index.tolist()
    test_indices = splits_list[0].X_test.index.tolist()
    n_train = len(train_indices)
    n_test = len(test_indices)

    dataset_hash = file_sha256(path)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    rows: List[Dict] = []
    best_model_name = None
    best_score = None
    best_pipe = None

    best_row: Dict = {}

    total_models = len(sel_models)
    progress = st.progress(0.0, text="Démarrage…")
    for idx, mname in enumerate(sel_models, start=1):
        start_ts = datetime.now()
        with st.expander(f"Entraînement: {mname}", expanded=False):
            status_txt = st.empty()
            try:
                status_txt.markdown("Préparation du pipeline…")
                model = get_model(mname, task, random_state=42)
                pipe = build_pipeline(preprocessor, model)
                cv_scores: Dict[str, float] = {}
                search_best_params = None
                search_space_used = None

                if do_search and search_iters > 0:
                    param_dist = get_param_distributions(mname, task)
                    if param_dist:
                        from sklearn.model_selection import RandomizedSearchCV
                        from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score

                        status_txt.markdown("Recherche d'hyperparamètres (RandomizedSearchCV avec métriques multiples)…")
                        search_space_used = param_dist
                        
                        # Définir les scorers pour avoir toutes les métriques pendant la recherche
                        if task == "regression":
                            scoring = {
                                'rmse': make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False),
                                'mae': make_scorer(mean_absolute_error, greater_is_better=False),
                                'r2': make_scorer(r2_score),
                            }
                            refit_metric = 'r2'  # Choisir le meilleur modèle selon R²
                        else:  # classification
                            scoring = {
                                'accuracy': make_scorer(accuracy_score),
                                'f1': make_scorer(f1_score, average='binary' if len(np.unique(split.y_train)) == 2 else 'macro'),
                            }
                            refit_metric = 'f1'  # Choisir le meilleur modèle selon F1
                        
                        search = RandomizedSearchCV(
                            estimator=pipe,
                            param_distributions=param_dist,
                            n_iter=int(search_iters),
                            cv=int(cv_folds),  # Même nombre de folds que la CV standard
                            scoring=scoring,
                            refit=refit_metric,  # Refitter avec la meilleure config selon cette métrique
                            n_jobs=-1,
                            random_state=42,
                            return_train_score=False,
                        )
                        search.fit(split.X_train, split.y_train)
                        pipe = search.best_estimator_
                        search_best_params = search.best_params_
                        
                        # Extraire les scores CV du meilleur modèle depuis cv_results_
                        status_txt.markdown("Extraction des métriques CV du meilleur modèle (sans double CV)…")
                        best_idx = search.best_index_
                        cv_results = search.cv_results_
                        
                        # Construire cv_scores avec mean, std et per-fold pour chaque métrique
                        cv_scores = {}
                        fold_scores_list = []
                        
                        if task == "regression":
                            metrics = ['rmse', 'mae', 'r2']
                        else:
                            metrics = ['accuracy', 'f1']
                        
                        # Pour chaque métrique, extraire mean, std et per-fold
                        for metric in metrics:
                            mean_key = f'mean_test_{metric}'
                            std_key = f'std_test_{metric}'
                            
                            if mean_key in cv_results:
                                mean_val = cv_results[mean_key][best_idx]
                                std_val = cv_results[std_key][best_idx]
                                
                                # Convertir les scores négatifs (rmse, mae) en positifs
                                if metric in ['rmse', 'mae']:
                                    mean_val = abs(mean_val)
                                
                                cv_scores[metric] = float(mean_val)
                                cv_scores[f'{metric}_std'] = float(std_val)
                        
                        # Extraire les scores per-fold pour chaque métrique
                        for fold_idx in range(int(cv_folds)):
                            fold_dict = {}
                            for metric in metrics:
                                fold_key = f'split{fold_idx}_test_{metric}'
                                if fold_key in cv_results:
                                    score = cv_results[fold_key][best_idx]
                                    # Convertir les scores négatifs en positifs
                                    if metric in ['rmse', 'mae']:
                                        score = abs(score)
                                    fold_dict[metric] = float(score)
                            if fold_dict:
                                fold_scores_list.append(fold_dict)
                        
                        cv_scores['_cv_scores_per_fold'] = fold_scores_list
                        
                        # Recréer les fold indices avec le même splitter pour reproductibilité
                        from sklearn.model_selection import KFold, StratifiedKFold
                        if task == "classification":
                            splitter = StratifiedKFold(n_splits=int(cv_folds), shuffle=True, random_state=42)
                            splits_iter = splitter.split(split.X_train, split.y_train)
                        else:
                            splitter = KFold(n_splits=int(cv_folds), shuffle=True, random_state=42)
                            splits_iter = splitter.split(split.X_train)
                        
                        cv_fold_indices = []
                        for train_idx, val_idx in splits_iter:
                            # Convertir en indices globaux
                            global_train_idx = [train_indices[i] for i in train_idx]
                            global_val_idx = [train_indices[i] for i in val_idx]
                            cv_fold_indices.append({
                                "train_indices": global_train_idx,
                                "val_indices": global_val_idx
                            })
                        
                        cv_scores['_cv_fold_indices'] = cv_fold_indices
                        cv_scores['search_best_score'] = float(search.best_score_)
                        cv_scores['search_n_iter'] = int(search_iters)
                        
                        # Sauvegarder TOUS les résultats de recherche pour analyse post-hoc
                        # Convertir les arrays numpy en listes pour la sérialisation JSON
                        search_cv_results_clean = {}
                        for key, value in cv_results.items():
                            if isinstance(value, np.ndarray):
                                # Convertir valeurs négatives (rmse, mae) en positives
                                if 'rmse' in key or 'mae' in key:
                                    search_cv_results_clean[key] = [abs(float(v)) if not np.isnan(v) else None for v in value]
                                else:
                                    search_cv_results_clean[key] = [float(v) if not np.isnan(v) else None for v in value]
                            elif isinstance(value, (list, tuple)):
                                search_cv_results_clean[key] = list(value)
                            else:
                                search_cv_results_clean[key] = value
                        cv_scores['_search_cv_results_all'] = search_cv_results_clean
                        
                    else:
                        status_txt.markdown("Pas d'espace de recherche pour ce modèle, validation croisée standard…")
                        cv_scores = cross_validate_pipeline(pipe, split.X_train, split.y_train, task=task, cv=int(cv_folds), random_state=42, train_indices=train_indices)
                else:
                    status_txt.markdown("Validation croisée en cours…")
                    cv_scores = cross_validate_pipeline(pipe, splits_list[0].X_train, splits_list[0].y_train, task=task, cv=int(cv_folds), random_state=42, train_indices=train_indices)

                status_txt.markdown("Entraînement final et évaluation test…")
                
                # Évaluations répétées ou unique
                test_results_per_repeat = []
                test_predictions = None  # stocke les prédictions du premier repeat pour analyse
                for rep_idx, split in enumerate(splits_list):
                    pipe_rep = get_model(mname, task, random_state=42)
                    pipe_rep = build_pipeline(preprocessor, pipe_rep)
                    pipe_rep.fit(split.X_train, split.y_train)
                    y_pred = pipe_rep.predict(split.X_test)
                    y_proba = None
                    if task == "classification" and hasattr(pipe_rep.named_steps["model"], "predict_proba"):
                        try:
                            y_proba = pipe_rep.predict_proba(split.X_test)
                        except Exception:
                            y_proba = None
                    test_metrics = compute_metrics(task, split.y_test.values, y_pred, y_proba)
                    test_results_per_repeat.append(test_metrics)
                    if rep_idx == 0:
                        # Garder les prédictions du premier split test pour la visualisation et l'export
                        try:
                            classes = getattr(pipe_rep.named_steps["model"], "classes_", None)
                            test_predictions = {
                                "y_true": split.y_test.tolist(),
                                "y_pred": y_pred.tolist(),
                                "y_proba": y_proba.tolist() if y_proba is not None else None,
                                "classes": classes.tolist() if classes is not None else None,
                                "index": split.X_test.index.tolist(),
                            }
                        except Exception:
                            test_predictions = None
                        best_pipe_candidate = pipe_rep
                
                # Moyenne et std des résultats test
                test_metrics_agg = {}
                if n_test_repeats > 1:
                    for metric_name in test_results_per_repeat[0].keys():
                        values = [r[metric_name] for r in test_results_per_repeat]
                        test_metrics_agg[metric_name] = float(np.mean(values))
                        test_metrics_agg[f"{metric_name}_std"] = float(np.std(values))
                else:
                    # Single test: add metrics + _std with NaN for consistency
                    for metric_name, val in test_results_per_repeat[0].items():
                        test_metrics_agg[metric_name] = val
                        test_metrics_agg[f"{metric_name}_std"] = np.nan
                
                # Récupérer les hyperparamètres du modèle entraîné (par défaut ou trouvés par recherche)
                model_params = best_pipe_candidate.named_steps["model"].get_params()
                
                row = {
                    "model": mname,
                    **{f"cv_{k}": v for k, v in cv_scores.items() if k not in ["_cv_scores_per_fold", "_cv_fold_indices"]},
                    **{f"test_{k}": v for k, v in test_metrics_agg.items()},
                }
                if search_best_params is not None:
                    row["search_best_params"] = search_best_params
                    row["model_params"] = model_params  # Enregistrer aussi les params finaux
                else:
                    row["model_params"] = model_params  # Enregistrer les params par défaut
                if search_space_used is not None:
                    row["search_space"] = search_space_used
                rows.append(row)
                
                # Ajouter les métadonnées techniques APRÈS l'ajout à rows (pour ne pas les inclure en CSV)
                row["_test_results_per_repeat"] = test_results_per_repeat
                row["_test_predictions"] = test_predictions
                row["_cv_scores_per_fold"] = cv_scores.pop("_cv_scores_per_fold", [])
                row["_cv_fold_indices"] = cv_scores.pop("_cv_fold_indices", [])
                row["_search_cv_results_all"] = cv_scores.pop("_search_cv_results_all", None)  # Tous les résultats de recherche

                # Sélection du best model (min RMSE ou max F1/accuracy)
                if task == "regression":
                    score = test_metrics_agg.get("rmse", np.inf)
                    better = best_score is None or score < best_score
                else:
                    score = test_metrics_agg.get("f1", test_metrics_agg.get("accuracy", 0.0))
                    better = best_score is None or score > best_score
                if better:
                    best_score = score
                    best_model_name = mname
                    best_pipe = best_pipe_candidate
                    best_row = row

                # Mesurer le temps d'exécution une fois
                elapsed = (datetime.now() - start_ts).total_seconds()

                # Sauvegarder tous les modèles si demandé
                preds_csv_path = None
                if save_all_models:
                    try:
                        meta_all = {
                            "data": str(path),
                            "target": sel_target,
                            "task": task,
                            "model_name": mname,
                            "test_metrics": test_metrics_agg,
                            "cv_metrics": {k.replace("cv_", ""): v for k, v in {f"cv_{mk}": mv for mk, mv in cv_scores.items() if mk != "_cv_scores_per_fold"}.items()},
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                            "fit_time_sec": float(elapsed),
                        }
                        stem = f"{path.stem}.{mname}.{task}"
                        model_path, json_path = save_model_and_meta(best_pipe_candidate, meta_all, MODELS_DIR, stem)
                        status_txt.markdown(f"✅ Modèle aussi sauvegardé dans models/")
                        # Exporter les prédictions test en CSV pour ce modèle (repeat #0)
                        if test_predictions is not None:
                            preds_df = pd.DataFrame({
                                "y_true": test_predictions.get("y_true", []),
                                "y_pred": test_predictions.get("y_pred", []),
                            })
                            if task == "classification" and test_predictions.get("y_proba") is not None:
                                proba_arr = np.array(test_predictions["y_proba"])
                                if proba_arr.ndim == 2 and proba_arr.shape[1] >= 2:
                                    # Ajouter proba de chaque classe
                                    classes = test_predictions.get("classes")
                                    if classes is None:
                                        classes = [f"class_{i}" for i in range(proba_arr.shape[1])]
                                    for idx_cls, cls_name in enumerate(classes):
                                        preds_df[f"proba_{cls_name}"] = proba_arr[:, idx_cls]
                                elif proba_arr.ndim == 1:
                                    preds_df["proba"] = proba_arr
                            preds_stem = f"preds_{path.stem}.{mname}.{task}.{run_id}.csv"
                            preds_csv_path = RESULTS_DIR / preds_stem
                            preds_df.to_csv(preds_csv_path, index=False)
                            status_txt.markdown(f"📄 Prédictions test exportées: {preds_csv_path.name}")
                    except Exception as e:
                        status_txt.markdown(f"⚠️ Erreur sauvegarde modèle: {str(e)[:50]}")

                # Enregistrer le temps d'exécution par modèle
                row["fit_time_sec"] = float(elapsed)
                if preds_csv_path is not None:
                    row["preds_csv"] = preds_csv_path.name
                status_txt.markdown(f"✅ Terminé en {elapsed:.1f}s — Test (mean): {test_metrics_agg}")
            except Exception as e:
                status_txt.markdown(f"❌ Erreur: {e}")
                st.error(f"Erreur sur {mname}: {e}")
        progress.progress(idx / total_models, text=f"{idx}/{total_models} modèles traités")

    if rows:
        # Sauvegarder best_pipe dans session_state pour persister après rerenderage (clés sans conflit avec widgets)
        if best_pipe is not None:
            st.session_state["_best_pipe"] = best_pipe
            st.session_state["_best_model_name"] = best_model_name
            st.session_state["_best_row"] = best_row
            st.session_state["_dataset_hash"] = dataset_hash
            st.session_state["_train_indices"] = train_indices
            st.session_state["_test_indices"] = test_indices
            st.session_state["_n_train"] = n_train
            st.session_state["_n_test"] = n_test
            st.session_state["_sel_target"] = sel_target
            st.session_state["_task"] = task
            st.session_state["_path"] = path
            st.session_state["_sel_models"] = sel_models
            st.session_state["_cv_folds"] = cv_folds
            st.session_state["_test_size"] = test_size
            st.session_state["_do_search"] = do_search
            st.session_state["_search_iters"] = search_iters
            
            # Sauvegarder automatiquement le meilleur modèle dans models/
            try:
                meta = {
                    "data": str(path),
                    "target": sel_target,
                    "task": task,
                    "models_tried": sel_models,
                    "cv_folds": int(cv_folds),
                    "test_size": test_size,
                    "do_search": bool(do_search),
                    "search_iters": int(search_iters) if do_search else 0,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "metrics_best": best_row,
                    "dataset_hash": dataset_hash,
                    "n_train": n_train,
                    "n_test": n_test,
                    "train_indices": train_indices,
                    "test_indices": test_indices,
                    "random_state_test": 42,
                    "random_state_cv": 42,
                    "fit_time_sec": float(best_row.get("fit_time_sec", np.nan)),
                }
                if best_row.get("search_best_params"):
                    meta["search_best_params"] = best_row.get("search_best_params")
                if best_row.get("search_space"):
                    meta["search_space"] = best_row.get("search_space")
                # Enregistrer tous les hyperparamètres du meilleur modèle (par défaut ou après recherche)
                if best_row.get("model_params"):
                    meta["model_params"] = best_row.get("model_params")
                stem = f"{path.stem}.{best_model_name}.{task}"
                model_path, json_path = save_model_and_meta(best_pipe, meta, MODELS_DIR, stem)
                st.success(f"✅ Meilleur modèle sauvegardé automatiquement:\n  - {model_path.name}\n  - {json_path.name}")
            except Exception as e:
                st.warning(f"⚠️ Impossible de sauvegarder le modèle automatiquement: {e}")
        
        df_res = pd.DataFrame(rows)
        st.subheader("Résultats")

        def _explain_column(col: str, task: str) -> str:
            # CV vs test
            prefix = "CV" if col.startswith("cv_") else ("Test" if col.startswith("test_") else "")
            name = col.split("_", 1)[1] if "_" in col else col
            better = "plus petit = mieux" if name in ("rmse", "mae") else "plus grand = mieux"
            if name in ("rmse", "mae", "r2", "accuracy", "f1", "roc_auc"):
                label = {
                    "rmse": "RMSE (erreur quadratique moyenne)",
                    "mae": "MAE (erreur absolue moyenne)",
                    "r2": "R² (variance expliquée)",
                    "accuracy": "Accuracy (proportion correcte)",
                    "f1": "F1 (balance précision/rappel)",
                    "roc_auc": "ROC AUC (aire sous la courbe)",
                }[name]
                where = f"{prefix} " if prefix else ""
                comp = "(moyenne des folds)" if prefix == "CV" else "(sur l'échantillon test)" if prefix == "Test" else ""
                return f"{where}{label} {comp} — {better}."
            if col == "model":
                return "Nom du modèle évalué."
            if col == "search_best_score":
                return "Meilleur score CV pendant la recherche d'hyperparamètres (scoring par défaut du modèle)."
            if col == "search_best_params":
                return "Hyperparamètres de la meilleure configuration trouvée."
            if col == "search_space":
                return "Espace de recherche exploré pour ce modèle."
            return ""

        # Tooltips via column_config (text columns to avoid Arrow mixed-type issues)
        colconf = {c: st.column_config.Column(help=_explain_column(c, task)) for c in df_res.columns}

        # Build a string-formatted display copy, then append footer row
        display_df = df_res.copy()
        try:
            for c in display_df.columns:
                if pd.api.types.is_float_dtype(display_df[c]):
                    display_df[c] = display_df[c].map(lambda x: f"{x:.4f}" if pd.notnull(x) else "")
                elif pd.api.types.is_integer_dtype(display_df[c]):
                    display_df[c] = display_df[c].map(lambda x: f"{int(x)}" if pd.notnull(x) else "")
        except Exception:
            pass
        
        # Fusionner mean et std pour les colonnes CV et Test
        display_df = display_df.astype(str)
        cols_to_drop = []
        for c in df_res.columns:
            if c.endswith("_std"):
                base_col = c[:-4]  # Enlever "_std"
                if base_col in display_df.columns:
                    # Fusionner base_col et base_col_std
                    try:
                        mean_val = pd.to_numeric(df_res[base_col], errors='coerce')
                        std_val = pd.to_numeric(df_res[c], errors='coerce')
                        display_df[base_col] = mean_val.map(lambda x: f"{x:.4f}" if pd.notnull(x) else "") + " ± " + std_val.map(lambda x: f"{x:.4f}" if pd.notnull(x) else "")
                        cols_to_drop.append(c)
                    except Exception:
                        pass
        
        # Enlever les colonnes _std du tableau affiché
        display_df = display_df.drop(columns=cols_to_drop, errors='ignore')
        
        footer = {c: _explain_column(c, task) for c in display_df.columns}
        footer["model"] = "Interprétation"
        display_df = pd.concat([display_df, pd.DataFrame([footer])], ignore_index=True)

        st.dataframe(display_df, width='stretch', column_config=colconf)
        csv_bytes = df_res.to_csv(index=False).encode("utf-8")
        st.download_button("Télécharger résultats (CSV)", csv_bytes, file_name="model_comparison.csv")

        # === Visualisations ===
        st.markdown("---")
        st.info("📊 Les graphiques (CV vs Test, gap, variance) sont disponibles dans l'app dédiée: `streamlit run apps/visualize_results.py`")

        # Auto-enregistrement des résultats et du contexte de run
        try:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            comp_csv_path = RESULTS_DIR / f"model_comparison_{run_id}.csv"
            comp_json_path = RESULTS_DIR / f"model_comparison_{run_id}.json"
            df_res.to_csv(comp_csv_path, index=False)
            run_meta = {
                "run_id": run_id,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "data": str(path),
                "dataset_hash": dataset_hash,
                "target": sel_target,
                "task": task,
                "test_size": test_size,
                "cv_folds": int(cv_folds),
                "random_state_test": 42,
                "random_state_cv": 42,
                "train_indices": train_indices,
                "test_indices": test_indices,
                "cv_fold_indices": rows[0].get("_cv_fold_indices", []) if rows else [],
                "do_search": bool(do_search),
                "search_iters": int(search_iters) if do_search else 0,
                "do_repeated": bool(do_repeated),
                "n_test_repeats": int(n_test_repeats),
                "models_tested": sel_models,
                "n_train": n_train,
                "n_test": n_test,
                "best_model": best_model_name,
                "best_row": best_row,
            }
            with open(comp_json_path, "w", encoding="utf8") as f:
                json.dump({"meta": run_meta, "results": rows}, f, ensure_ascii=False, indent=2)
            st.success(f"Résultats auto-enregistrés dans results: {comp_csv_path.name} / {comp_json_path.name}")
            # Conserver en mémoire pour affichage même après un rerender Streamlit
            st.session_state["last_run"] = {
                "rows": rows,
                "meta": run_meta,
                "csv": comp_csv_path.name,
                "json": comp_json_path.name,
            }
        except Exception as e:
            st.warning(f"Impossible d'enregistrer automatiquement les résultats: {e}")
    else:
        st.info("Aucun résultat à afficher.")
