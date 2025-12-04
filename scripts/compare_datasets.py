#!/usr/bin/env python3
"""Comparaison rapide de deux jeux de données présents dans `data/`.

Génère `docs/COMPARAISON_DATASETS.md` avec des statistiques par jeu et
une recommandation succincte sur le jeu à privilégier pour l'entraînement.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'
OUT_MD = ROOT / 'docs' / 'COMPARAISON_DATASETS.md'


def load_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"Échec de lecture {path} : {e}")
        raise


def summarize(df: pd.DataFrame) -> dict:
    numeric = df.select_dtypes(include='number')
    categorical = df.select_dtypes(include=['object', 'category'])
    summary = {
        'rows': len(df),
        'cols': df.shape[1],
        'num_numeric': numeric.shape[1],
        'num_categorical': categorical.shape[1],
        'missing_total': int(df.isna().sum().sum()),
        'missing_perc': float(df.isna().sum().sum() / (df.shape[0]*df.shape[1]) * 100),
        'numeric_preview': numeric.describe().loc[['min','mean','50%','max']].to_dict() if not numeric.empty else {},
        'top_categoricals': {c: df[c].nunique() for c in categorical.columns[:5]}
    }
    return summary


def find_target_column(df: pd.DataFrame) -> str | None:
    # heuristics: common names
    candidates = ['price', 'Price', 'sale_price', 'SalePrice', 'precio']
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: choose numeric column with name containing 'price' (case-insensitive)
    for c in df.select_dtypes(include='number').columns:
        if 'price' in c.lower() or 'precio' in c.lower():
            return c
    # else None
    return None


def render_md(info_a: dict, info_b: dict, name_a: str, name_b: str, recommend: str) -> str:
    md = []
    md.append(f"# Comparaison des jeux de données\n")
    md.append(f"Fichiers comparés: `{name_a}` et `{name_b}`\n")

    def section(name, info):
        s = [f"## {name}\n"]
        s.append(f"- Lignes : {info['rows']}\n")
        s.append(f"- Colonnes : {info['cols']}\n")
        s.append(f"- Colonnes numériques : {info['num_numeric']}\n")
        s.append(f"- Colonnes catégorielles : {info['num_categorical']}\n")
        s.append(f"- Valeurs manquantes (total) : {info['missing_total']} (~{info['missing_perc']:.2f}%)\n")
        if info['numeric_preview']:
            s.append("- Aperçu statistiques (quelques colonnes numériques) :\n")
            for col, stats in list(info['numeric_preview'].items())[:5]:
                s.append(f"  - {col}: min={stats['min']:.3g}, med={stats['50%']:.3g}, mean={stats['mean']:.3g}, max={stats['max']:.3g}\n")
        if info['top_categoricals']:
            s.append("- Exemples de colonnes catégorielles (nombre valeurs uniques) :\n")
            for c, n in info['top_categoricals'].items():
                s.append(f"  - {c}: {n}\n")
        return '\n'.join(s)

    md.append(section(name_a, info_a))
    md.append(section(name_b, info_b))

    md.append("## Comparaison rapide\n")
    md.append(f"- `{name_a}` : {info_a['rows']} lignes, {info_a['cols']} colonnes, {info_a['missing_total']} valeurs manquantes.\n")
    md.append(f"- `{name_b}` : {info_b['rows']} lignes, {info_b['cols']} colonnes, {info_b['missing_total']} valeurs manquantes.\n")

    md.append("## Recommandation\n")
    md.append(recommend + "\n")

    md.append("---\n")
    md.append("_Généré automatiquement par `scripts/compare_datasets.py`._\n")
    return '\n'.join(md)


def build_recommendation(info_a, info_b, name_a, name_b, target_a, target_b) -> str:
    reasons = []
    # prefer dataset with target column
    if target_a and not target_b:
        reasons.append(f"`{name_a}` contient une colonne cible identifiée (`{target_a}`), alors que `{name_b}` n'en a pas clairement.")
    if target_b and not target_a:
        reasons.append(f"`{name_b}` contient une colonne cible identifiée (`{target_b}`), alors que `{name_a}` n'en a pas clairement.")
    # prefer more rows
    if info_a['rows'] > info_b['rows'] * 1.2:
        reasons.append(f"`{name_a}` est significativement plus grand ({info_a['rows']} lignes) que `{name_b}` ({info_b['rows']} lignes).")
    elif info_b['rows'] > info_a['rows'] * 1.2:
        reasons.append(f"`{name_b}` est significativement plus grand ({info_b['rows']} lignes) que `{name_a}` ({info_a['rows']} lignes).")
    # prefer fewer missing values
    if info_a['missing_total'] < info_b['missing_total'] * 0.7:
        reasons.append(f"`{name_a}` a moins de valeurs manquantes (total {info_a['missing_total']}) que `{name_b}` ({info_b['missing_total']}).")
    elif info_b['missing_total'] < info_a['missing_total'] * 0.7:
        reasons.append(f"`{name_b}` a moins de valeurs manquantes (total {info_b['missing_total']}) que `{name_a}` ({info_a['missing_total']}).")

    # default tie-breaker: prefer dataset local (Madrid) if present (useful for geo-specific modelling)
    pref = None
    if 'Madrid' in name_a or 'madrid' in name_a.lower():
        pref = name_a
    if 'Madrid' in name_b or 'madrid' in name_b.lower():
        pref = name_b

    if not reasons:
        if pref:
            return f"Aucune différence claire sur la qualité ; je recommande toutefois d'utiliser `{pref}` si vous visez une application centrée sur la zone géographique correspondante." + "\n\nRaisons: données comparables." 
        return "Aucune différence claire et significative trouvée automatiquement. Choisissez selon la disponibilité des métadonnées et la pertinence géographique pour votre projet."

    # build final recommendation
    rec = "Nous recommandons d'utiliser le jeu de données qui suit, basé sur les critères ci-dessus:\n"
    # choose winner based on simple score
    score_a = 0
    score_b = 0
    if target_a: score_a += 2
    if target_b: score_b += 2
    if info_a['rows'] > info_b['rows']: score_a += 1
    if info_b['rows'] > info_a['rows']: score_b += 1
    if info_a['missing_total'] < info_b['missing_total']: score_a += 1
    if info_b['missing_total'] < info_a['missing_total']: score_b += 1

    winner = name_a if score_a >= score_b else name_b
    rec += f"**Choix** : `{winner}`.\n\n"
    rec += "Raisons succinctes :\n"
    for r in reasons:
        rec += f"- {r}\n"
    if pref and winner != pref:
        rec += f"_Note_: même si `{pref}` est géographiquement pertinente, le score automatique a désigné `{winner}`. Prenez en compte l'objectif géographique final.\n"
    return rec


def main() -> int:
    a = DATA_DIR / 'houses_Madrid.csv'
    b = DATA_DIR / 'kc_house_data.csv'
    if not a.exists() or not b.exists():
        print(f"Attendu: {a} et {b} dans data/. Trouvés: {a.exists()}, {b.exists()}")
        return 2

    df_a = load_csv(a)
    df_b = load_csv(b)

    info_a = summarize(df_a)
    info_b = summarize(df_b)

    target_a = find_target_column(df_a)
    target_b = find_target_column(df_b)

    recommend = build_recommendation(info_a, info_b, a.name, b.name, target_a, target_b)
    md = render_md(info_a, info_b, a.name, b.name, recommend)

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(md, encoding='utf-8')
    print(f"Wrote {OUT_MD}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
