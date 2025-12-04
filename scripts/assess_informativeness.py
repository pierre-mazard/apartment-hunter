#!/usr/bin/env python3
"""Évalue l'«informativeness» de deux jeux de données pour la prédiction de prix.

Génère `docs/INFORMATIVENESS_REPORT.md` contenant :
- la colonne cible détectée (si présente)
- la liste des prédicteurs numériques candidats
- le taux de valeurs manquantes sur variables clés
- si la cible est présente : corrélations et information mutuelle (top features)
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'
OUT_MD = ROOT / 'docs' / 'INFORMATIVENESS_REPORT.md'


def load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def detect_target(df: pd.DataFrame) -> str | None:
    candidates = ['price', 'Price', 'sale_price', 'SalePrice', 'precio']
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.select_dtypes(include='number').columns:
        if 'price' in c.lower() or 'precio' in c.lower():
            return c
    return None


def candidate_predictors(df: pd.DataFrame) -> list[str]:
    # numeric predictors excluding id and target
    num = list(df.select_dtypes(include='number').columns)
    cand = [c for c in num if c.lower() not in ('id', 'index')]
    return cand


def key_feature_missingness(df: pd.DataFrame, candidates: list[str]) -> list[tuple]:
    # return top candidate features and their missing %
    rows = len(df)
    res = []
    for c in candidates:
        miss = df[c].isna().sum()
        res.append((c, miss, miss / rows * 100))
    res.sort(key=lambda x: x[2])
    return res


def top_correlations(df: pd.DataFrame, target: str, topk=10):
    numeric = df.select_dtypes(include='number')
    if target not in numeric.columns:
        return []
    corr = numeric.corr()[target].abs().sort_values(ascending=False)
    corr = corr.drop(labels=[target], errors='ignore')
    return corr.head(topk).to_dict()


def top_mutual_info(df: pd.DataFrame, target: str, topk=10):
    numeric = df.select_dtypes(include='number')
    if target not in numeric.columns:
        return []
    X = numeric.drop(columns=[target]).fillna(0)
    y = numeric[target].fillna(0)
    try:
        mi = mutual_info_regression(X, y, random_state=0)
        mi_s = pd.Series(mi, index=X.columns).sort_values(ascending=False).head(topk)
        return mi_s.to_dict()
    except Exception:
        return {}


def analyze(path: Path) -> dict:
    df = load(path)
    tgt = detect_target(df)
    preds = candidate_predictors(df)
    missing = key_feature_missingness(df, preds[:20])
    corr = top_correlations(df, tgt) if tgt else {}
    mi = top_mutual_info(df, tgt) if tgt else {}
    # geo info
    geo = any(c.lower() in ('lat', 'latitude', 'lon', 'lng', 'longitude') for c in df.columns)
    text_cols = [c for c in df.select_dtypes(include='object').columns if df[c].nunique() > 50]
    return {
        'name': path.name,
        'rows': len(df),
        'cols': df.shape[1],
        'target': tgt,
        'num_numeric': df.select_dtypes(include='number').shape[1],
        'num_categorical': df.select_dtypes(include='object').shape[1],
        'missing_total': int(df.isna().sum().sum()),
        'missing_pct': float(df.isna().sum().sum() / (df.shape[0]*df.shape[1]) * 100),
        'top_preds_sample': preds[:20],
        'missing_on_top_preds': missing,
        'geo_fields': geo,
        'text_rich_columns_sample': text_cols[:10],
        'top_correlations': corr,
        'top_mutual_info': mi,
    }


def render(mdpath: Path, a: dict, b: dict) -> None:
    lines = []
    lines.append('# Rapport d\'informativeness des jeux de données\n')
    for info in (a, b):
        lines.append(f"## {info['name']}\n")
        lines.append(f"- Lignes: {info['rows']}")
        lines.append(f"- Colonnes: {info['cols']}")
        lines.append(f"- Cible détectée: {info['target']}")
        lines.append(f"- Colonnes numériques: {info['num_numeric']}")
        lines.append(f"- Colonnes catégorielles: {info['num_categorical']}")
        lines.append(f"- Valeurs manquantes (total): {info['missing_total']} (~{info['missing_pct']:.2f}%)")
        lines.append(f"- Présence info géographique (lat/lon): {info['geo_fields']}")
        lines.append(f"- Exemples de colonnes textuelles riches (n>50 uniq): {', '.join(info['text_rich_columns_sample'] or ['(none)'])}")
        lines.append('\n- Exemples de prédicteurs numériques (échantillon):')
        for p in info['top_preds_sample'][:10]:
            lines.append(f"  - {p}")
        lines.append('\n- Missing% on top predictors (sample):')
        for c, miss, pct in info['missing_on_top_preds'][:10]:
            lines.append(f"  - {c}: {miss} ({pct:.2f}%)")
        if info['top_correlations']:
            lines.append('\n- Top corr. absolues avec la cible:')
            for c, v in info['top_correlations'].items():
                lines.append(f"  - {c}: {v:.3f}")
        if info['top_mutual_info']:
            lines.append('\n- Top mutual info:')
            for c, v in info['top_mutual_info'].items():
                lines.append(f"  - {c}: {v:.4f}")
        lines.append('\n---\n')

    # short recommendation heuristic
    def score(info):
        s = 0
        if info['target']:
            s += 3
        s += min(3, info['num_numeric'] / 5)
        miss_mean = np.mean([pct for (_, _, pct) in info['missing_on_top_preds']]) if info['missing_on_top_preds'] else 100
        s += max(0, 2 - miss_mean / 50)
        if info['geo_fields']:
            s += 1
        return s

    sa = score(a)
    sb = score(b)
    winner = a['name'] if sa >= sb else b['name']
    lines.append('## Recommandation automatique\n')
    lines.append(f'- Score {a["name"]}: {sa:.2f}')
    lines.append(f'- Score {b["name"]}: {sb:.2f}')
    lines.append(f'**Choix recommandé**: `{winner}`\n')
    lines.append('\n_Notes_: cette recommandation est heuristique — examinez les corrélations, la présence d\'une colonne cible, et la qualité des features avant décision finale._\n')

    mdpath.parent.mkdir(parents=True, exist_ok=True)
    mdpath.write_text('\n'.join(lines), encoding='utf-8')


def main() -> int:
    a = DATA_DIR / 'houses_Madrid.csv'
    b = DATA_DIR / 'kc_house_data.csv'
    if not a.exists() or not b.exists():
        print('Les fichiers attendus manquent dans data/')
        return 2
    ia = analyze(a)
    ib = analyze(b)
    render(OUT_MD, ia, ib)
    print(f'Wrote {OUT_MD}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
