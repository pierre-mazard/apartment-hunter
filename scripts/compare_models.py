#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import pandas as pd

from src.ml.pipelines import (
    build_pipeline,
    build_preprocessor,
    cross_validate_pipeline,
    infer_feature_sets,
    get_model,
    split_data,
    compute_metrics,
    get_param_distributions,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare multiple models on a cleaned dataset")
    p.add_argument("--data", required=True, help="Path to cleaned CSV")
    p.add_argument("--target", required=True, help="Target column name")
    p.add_argument("--task", choices=["regression", "classification"], required=True)
    p.add_argument("--models", nargs="+", required=True, help="Model names (space-separated)")
    p.add_argument("--cv", type=int, default=5, help="Number of CV folds")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--out-csv", default="results/model_comparison.csv", help="Output CSV for summary")
    p.add_argument("--search-iters", type=int, default=0, help="RandomizedSearchCV iterations per model (0 to skip)")
    p.add_argument("--search-cv", type=int, default=3, help="CV folds for hyperparam search")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(Path(args.data))
    if args.target not in df.columns:
        raise SystemExit(f"Target '{args.target}' not found")

    num_cols, cat_cols = infer_feature_sets(df, args.target)
    preprocessor = build_preprocessor(num_cols, cat_cols)

    # Single train/test split for fair comparison
    split = split_data(df, args.target, test_size=args.test_size, random_state=args.random_state, task=args.task)

    rows: List[dict] = []
    for mname in args.models:
        try:
            model = get_model(mname, args.task, random_state=args.random_state)
            pipe = build_pipeline(preprocessor, model)

            # Optional hyperparameter search
            if args.search_iters and args.search_iters > 0:
                from sklearn.model_selection import RandomizedSearchCV

                param_dist = get_param_distributions(mname, args.task)
                if param_dist:
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
                    cv_scores = {"search_best_score": float(search.best_score_)}
                else:
                    cv_scores = cross_validate_pipeline(pipe, split.X_train, split.y_train, task=args.task, cv=args.cv, random_state=args.random_state)
            else:
                cv_scores = cross_validate_pipeline(pipe, split.X_train, split.y_train, task=args.task, cv=args.cv, random_state=args.random_state)

            pipe.fit(split.X_train, split.y_train)
            y_pred = pipe.predict(split.X_test)
            y_proba = None
            if args.task == "classification" and hasattr(pipe.named_steps["model"], "predict_proba"):
                try:
                    y_proba = pipe.predict_proba(split.X_test)
                except Exception:
                    y_proba = None
            test_metrics = compute_metrics(args.task, split.y_test.values, y_pred, y_proba)
            row = {"model": mname, **{f"cv_{k}": v for k, v in cv_scores.items()}, **{f"test_{k}": v for k, v in test_metrics.items()}}
            rows.append(row)
            print(f"Model {mname}: CV {cv_scores} | Test {test_metrics}")
        except Exception as e:
            print(f"Model {mname}: error {e}")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    # Compute common headers
    headers = sorted(set().union(*[set(r.keys()) for r in rows]))
    with open(out_csv, "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("Saved comparison:", out_csv)


if __name__ == "__main__":
    main()
