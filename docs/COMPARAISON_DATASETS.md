# Comparaison des jeux de données

Fichiers comparés: `houses_Madrid.csv` et `kc_house_data.csv`

## houses_Madrid.csv

- Lignes : 21742

- Colonnes : 58

- Colonnes numériques : 23

- Colonnes catégorielles : 30

- Valeurs manquantes (total) : 542327 (~43.01%)

- Aperçu statistiques (quelques colonnes numériques) :

  - Unnamed: 0: min=0, med=1.09e+04, mean=1.09e+04, max=2.17e+04

  - id: min=1, med=1.09e+04, mean=1.09e+04, max=2.17e+04

  - sq_mt_built: min=13, med=100, mean=147, max=999

  - sq_mt_useful: min=1, med=79, mean=103, max=998

  - n_rooms: min=0, med=3, mean=3.01, max=24

- Exemples de colonnes catégorielles (nombre valeurs uniques) :

  - title: 10736

  - subtitle: 146

  - raw_address: 9666

  - street_name: 6177

  - street_number: 420

## kc_house_data.csv

- Lignes : 21613

- Colonnes : 21

- Colonnes numériques : 20

- Colonnes catégorielles : 1

- Valeurs manquantes (total) : 0 (~0.00%)

- Aperçu statistiques (quelques colonnes numériques) :

  - id: min=1e+06, med=3.9e+09, mean=4.58e+09, max=9.9e+09

  - price: min=7.5e+04, med=4.5e+05, mean=5.4e+05, max=7.7e+06

  - bedrooms: min=0, med=3, mean=3.37, max=33

  - bathrooms: min=0, med=2.25, mean=2.11, max=8

  - sqft_living: min=290, med=1.91e+03, mean=2.08e+03, max=1.35e+04

- Exemples de colonnes catégorielles (nombre valeurs uniques) :

  - date: 372

## Comparaison rapide

- `houses_Madrid.csv` : 21742 lignes, 58 colonnes, 542327 valeurs manquantes.

- `kc_house_data.csv` : 21613 lignes, 21 colonnes, 0 valeurs manquantes.

## Recommandation

Nous recommandons d'utiliser le jeu de données qui suit, basé sur les critères ci-dessus:
**Choix** : `houses_Madrid.csv`.

Raisons succinctes :
- `kc_house_data.csv` a moins de valeurs manquantes (total 0) que `houses_Madrid.csv` (542327).


---

_Généré automatiquement par `scripts/compare_datasets.py`._
