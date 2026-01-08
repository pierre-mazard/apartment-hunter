#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import r2_score

from src.ml.pipelines import (
    build_pipeline,
    build_preprocessor,
    compute_metrics,
    cross_validate_pipeline,
    infer_feature_sets,
    save_model_and_meta,
    split_data,
    get_model,
    get_param_distributions,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a model on a cleaned dataset")
    p.add_argument("--data", required=True, help="Path to cleaned CSV")
    p.add_argument("--target", required=True, help="Target column name")
    p.add_argument("--task", choices=["regression", "classification"], required=True)
    p.add_argument("--model", required=True, help="Model name (e.g., linear, random_forest, gradient_boosting)")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--cv", type=int, default=0, help="Number of CV folds (0 to skip)")
    p.add_argument("--search-iters", type=int, default=0, help="RandomizedSearchCV iterations (0 to skip)")
    p.add_argument("--search-cv", type=int, default=3, help="CV folds for hyperparam search")
    p.add_argument("--out-dir", default="models", help="Output directory for model artifacts")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    out_dir = Path(args.out_dir)

    df = pd.read_csv(data_path)
    # Ensure target present
    if args.target not in df.columns:
        raise SystemExit(f"Target '{args.target}' not in dataset columns")

    # Feature sets and preprocessing
    num_cols, cat_cols = infer_feature_sets(df, args.target)
    preprocessor = build_preprocessor(num_cols, cat_cols)
    model = get_model(args.model, args.task, random_state=args.random_state)
    pipe = build_pipeline(preprocessor, model)

    # Split
    split = split_data(df, args.target, test_size=args.test_size, random_state=args.random_state, task=args.task)

    # Optional CV on training set
    cv_scores = {}
    if args.cv and args.cv > 1:
        cv_scores = cross_validate_pipeline(pipe, split.X_train, split.y_train, task=args.task, cv=args.cv, random_state=args.random_state)
        print("CV metrics:", cv_scores)

    # Optional hyperparameter search (RandomizedSearchCV)
    if args.search_iters and args.search_iters > 0:
        param_dist = get_param_distributions(args.model, args.task)
        if param_dist:
            from sklearn.model_selection import RandomizedSearchCV

            search = RandomizedSearchCV(
                estimator=pipe,
                param_distributions=param_dist,
                n_iter=args.search_iters,
                cv=args.search_cv,
                n_jobs=-1,
                random_state=args.random_state,
            )
            search.fit(split.X_train, split.y_train)
            pipe = search.best_estimator_
            cv_scores = search.best_score_ if hasattr(search, "best_score_") else cv_scores
            print("Search best params:", search.best_params_)
        else:
            print("No search space for this model; skipping hyperparameter search.")

    # Fit and evaluate
    pipe.fit(split.X_train, split.y_train)
    y_pred = pipe.predict(split.X_test)
    # proba if available (classification)
    y_proba = None
    try:
        if args.task == "classification" and hasattr(pipe.named_steps["model"], "predict_proba"):
            y_proba = pipe.predict_proba(split.X_test)
    except Exception:
        y_proba = None
    test_metrics = compute_metrics(args.task, split.y_test.values, y_pred, y_proba)
    print("Test metrics:", test_metrics)

    # Save model and metadata
    stem = f"{data_path.stem}.{args.model}.{args.task}"
    meta = {
        "data": str(data_path),
        "target": args.target,
        "task": args.task,
        "model": args.model,
        "random_state": args.random_state,
        "test_size": args.test_size,
        "cv": args.cv,
        "cv_scores": cv_scores,
        "test_metrics": test_metrics,
        "feature_counts": {"numeric": len(num_cols), "categorical": len(cat_cols)},
    }
    model_path, json_path = save_model_and_meta(pipe, meta, out_dir, stem)
    print("Saved:", model_path, json_path)


if __name__ == "__main__":
    main()
