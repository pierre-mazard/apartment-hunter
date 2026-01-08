from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier


@dataclass
class SplitResult:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def infer_feature_sets(df: pd.DataFrame, target: str) -> Tuple[List[str], List[str]]:
    """Infer numeric and categorical feature column names, excluding target.
    - Numeric: pandas 'number' dtypes
    - Categorical: object/category/string
    """
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in dataframe columns")
    feature_cols = [c for c in df.columns if c != target]
    df_features = df[feature_cols]
    num_cols = df_features.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df_features.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    return num_cols, cat_cols


def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ]
    )
    return preprocessor


def get_model(model_name: str, task: str, random_state: int = 42):
    task = task.lower()
    model_name = model_name.lower()
    if task == "regression":
        if model_name in ("linear", "linreg", "linear_regression"):
            return LinearRegression()
        if model_name in ("rf", "random_forest"):
            return RandomForestRegressor(random_state=random_state)
        if model_name in ("gbr", "gradient_boosting"):
            return GradientBoostingRegressor(random_state=random_state)
        raise ValueError(f"Unknown regression model '{model_name}'")
    elif task == "classification":
        if model_name in ("logreg", "logistic", "logistic_regression"):
            return LogisticRegression(max_iter=1000, random_state=random_state)
        if model_name in ("rf", "random_forest"):
            return RandomForestClassifier(random_state=random_state)
        if model_name in ("gbc", "gradient_boosting"):
            return GradientBoostingClassifier(random_state=random_state)
        raise ValueError(f"Unknown classification model '{model_name}'")
    else:
        raise ValueError(f"Unknown task '{task}', expected 'regression' or 'classification'")


def get_param_distributions(model_name: str, task: str) -> Optional[Dict[str, list]]:
    """Return a small hyperparameter search space for supported models.
    Intended for quick RandomizedSearchCV; keep spaces small to stay fast.
    """
    task = task.lower()
    model_name = model_name.lower()
    if task == "regression":
        if model_name in ("rf", "random_forest"):
            return {
                "model__n_estimators": [100, 200, 400],
                "model__max_depth": [None, 5, 10, 20],
                "model__min_samples_split": [2, 5, 10],
            }
        if model_name in ("gbr", "gradient_boosting"):
            return {
                "model__n_estimators": [100, 200, 400],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__max_depth": [2, 3, 4],
            }
        # linear regression has no meaningful grid here
        return None
    else:
        if model_name in ("logreg", "logistic", "logistic_regression"):
            return {
                "model__C": [0.1, 0.5, 1.0, 2.0, 5.0],
                "model__penalty": ["l2"],
                "model__solver": ["lbfgs", "liblinear"],
            }
        if model_name in ("rf", "random_forest"):
            return {
                "model__n_estimators": [100, 200, 400],
                "model__max_depth": [None, 5, 10, 20],
                "model__min_samples_split": [2, 5, 10],
            }
        if model_name in ("gbc", "gradient_boosting"):
            return {
                "model__n_estimators": [100, 200, 400],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__max_depth": [2, 3, 4],
            }
        return None


def build_pipeline(preprocessor: ColumnTransformer, model) -> Pipeline:
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def split_data(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    random_state: int = 42,
    task: str = "regression",
) -> SplitResult:
    X = df.drop(columns=[target])
    y = df[target]
    stratify = y if task.lower() == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    return SplitResult(X_train, X_test, y_train, y_test)


def compute_metrics(task: str, y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    task = task.lower()
    if task == "regression":
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        return {"rmse": rmse, "mae": mae, "r2": r2}
    else:
        acc = float(accuracy_score(y_true, y_pred))
        # If multi-class, use macro F1
        f1 = float(f1_score(y_true, y_pred, average="binary" if len(np.unique(y_true)) == 2 else "macro"))
        metrics = {"accuracy": acc, "f1": f1}
        if y_proba is not None:
            try:
                # Use probability for positive class (binary) or multi-class OVR
                if y_proba.ndim == 2 and y_proba.shape[1] > 1:
                    auc = float(roc_auc_score(y_true, y_proba, multi_class="ovr"))
                else:
                    auc = float(roc_auc_score(y_true, y_proba))
                metrics["roc_auc"] = auc
            except Exception:
                pass
        return metrics


def cross_validate_pipeline(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    cv: int = 5,
    random_state: int = 42,
    train_indices: Optional[List[int]] = None,
) -> Dict:
    """
    Cross-validate pipeline and return both aggregated metrics (mean/std) and individual fold scores.
    Returns dict with:
      - Aggregated: rmse, rmse_std, r2, r2_std, mae, mae_std (or accuracy, f1 for classification)
      - Per-fold: _cv_scores_per_fold = [{rmse: ..., r2: ..., mae: ...}, ...]
      - CV split indices: _cv_fold_indices = [{'train_indices': [...], 'val_indices': [...]}, ...]
        (indices globaux du dataset original si train_indices fourni, sinon indices locaux)
    """
    if task.lower() == "classification":
        splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        scoring = ["accuracy", "f1"]
    else:
        splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
        scoring = ["neg_root_mean_squared_error", "r2", "neg_mean_absolute_error"]
    
    # Capture les indices des folds pour reproductibilité
    cv_fold_indices = []
    for train_idx, val_idx in splitter.split(X, y if task.lower() == "classification" else None):
        # Si train_indices fourni, convertir les indices locaux en indices globaux du dataset original
        if train_indices is not None:
            global_train_idx = [train_indices[i] for i in train_idx]
            global_val_idx = [train_indices[i] for i in val_idx]
        else:
            global_train_idx = train_idx.tolist()
            global_val_idx = val_idx.tolist()
        
        cv_fold_indices.append({
            "train_indices": global_train_idx,
            "val_indices": global_val_idx
        })
    
    cv_res = cross_validate(pipeline, X, y, cv=splitter, scoring=scoring, n_jobs=-1)
    
    # Aggregate metrics (mean/std)
    out: Dict[str, float] = {}
    fold_scores: List[Dict[str, float]] = []
    
    # Map scoring keys to normalized names
    score_mapping = {
        "neg_root_mean_squared_error": "rmse",
        "neg_mean_absolute_error": "mae",
        "r2": "r2",
        "accuracy": "accuracy",
        "f1": "f1",
    }
    
    # Build per-fold scores list
    num_folds = len(cv_res.get("test_neg_root_mean_squared_error", cv_res.get("test_accuracy", [])))
    for fold_idx in range(num_folds):
        fold_dict = {}
        for k, v in cv_res.items():
            if k.startswith("test_"):
                name = k.replace("test_", "")
                arr = v
                fold_val = arr[fold_idx]
                if name in ("neg_root_mean_squared_error", "neg_mean_absolute_error"):
                    fold_val = -fold_val
                norm_name = score_mapping.get(name, name)
                fold_dict[norm_name] = float(fold_val)
        fold_scores.append(fold_dict)
    
    # Aggregate
    for k, v in cv_res.items():
        if k.startswith("test_"):
            name = k.replace("test_", "")
            arr = v
            if name in ("neg_root_mean_squared_error", "neg_mean_absolute_error"):
                arr = -arr
            norm_name = score_mapping.get(name, name)
            out[norm_name] = float(np.mean(arr))
            out[f"{norm_name}_std"] = float(np.std(arr))
    
    out["_cv_scores_per_fold"] = fold_scores
    out["_cv_fold_indices"] = cv_fold_indices
    return out


def save_model_and_meta(
    pipeline: Pipeline,
    meta: Dict,
    out_dir: Path,
    stem: str,
) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    model_path = out_dir / f"{stem}.{ts}.joblib"
    json_path = out_dir / f"{stem}.{ts}.meta.json"
    joblib.dump(pipeline, model_path)
    with open(json_path, "w", encoding="utf8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return model_path, json_path
