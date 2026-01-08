#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src.ml.pipelines import compute_metrics, infer_feature_sets, build_preprocessor, build_pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a saved model on a dataset")
    p.add_argument("--data", required=True, help="Path to cleaned CSV")
    p.add_argument("--target", required=True, help="Target column name")
    p.add_argument("--task", choices=["regression", "classification"], required=True)
    p.add_argument("--model-path", required=True, help="Path to saved .joblib model")
    p.add_argument("--out-json", default="", help="Optional path to save metrics JSON")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    model_path = Path(args.model_path)

    df = pd.read_csv(data_path)
    if args.target not in df.columns:
        raise SystemExit(f"Target '{args.target}' not in dataset columns")

    # Load model
    pipe = joblib.load(model_path)

    # Evaluate
    X = df.drop(columns=[args.target])
    y = df[args.target]
    y_pred = pipe.predict(X)
    y_proba = None
    try:
        if args.task == "classification" and hasattr(pipe.named_steps["model"], "predict_proba"):
            y_proba = pipe.predict_proba(X)
    except Exception:
        y_proba = None

    metrics = compute_metrics(args.task, y.values, y_pred, y_proba)
    print("Metrics:", metrics)

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print("Saved metrics:", out_path)


if __name__ == "__main__":
    main()
