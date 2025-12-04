# Rapport d'informativeness des jeux de données

## houses_Madrid.csv

- Lignes: 21742
- Colonnes: 58
- Cible détectée: rent_price
- Colonnes numériques: 23
- Colonnes catégorielles: 30
- Valeurs manquantes (total): 542327 (~43.01%)
- Présence info géographique (lat/lon): True
- Exemples de colonnes textuelles riches (n>50 uniq): title, subtitle, raw_address, street_name, street_number, neighborhood_id

- Exemples de prédicteurs numériques (échantillon):
  - Unnamed: 0
  - sq_mt_built
  - sq_mt_useful
  - n_rooms
  - n_bathrooms
  - n_floors
  - sq_mt_allotment
  - latitude
  - longitude
  - portal

-- Pourcentage de valeurs manquantes sur les principaux prédicteurs (échantillon) :
  - Unnamed: 0: 0 (0.00%)
  - n_rooms: 0 (0.00%)
  - rent_price: 0 (0.00%)
  - buy_price: 0 (0.00%)
  - buy_price_by_area: 0 (0.00%)
  - n_bathrooms: 16 (0.07%)
  - sq_mt_built: 126 (0.58%)
  - built_year: 11742 (54.01%)
  - sq_mt_useful: 13514 (62.16%)
  - n_floors: 20305 (93.39%)

- Top corr. absolues avec la cible:
  - buy_price: 0.468
  - sq_mt_built: 0.235
  - sq_mt_useful: 0.221
  - n_bathrooms: 0.197
  - n_rooms: 0.158
  - buy_price_by_area: 0.151
  - sq_mt_allotment: 0.120
  - Unnamed: 0: 0.020
  - id: 0.020
  - n_floors: 0.014

- Top — information mutuelle :
  - buy_price: 5.6769
  - buy_price_by_area: 1.2156
  - sq_mt_built: 0.9456
  - n_bathrooms: 0.5551
  - id: 0.4469
  - Unnamed: 0: 0.4463
  - n_rooms: 0.3468
  - sq_mt_useful: 0.2600
  - built_year: 0.1609
  - sq_mt_allotment: 0.1157

---

## kc_house_data.csv

- Lignes: 21613
- Colonnes: 21
- Cible détectée: price
- Colonnes numériques: 20
- Colonnes catégorielles: 1
- Valeurs manquantes (total): 0 (~0.00%)
- Présence info géographique (lat/lon): True
- Exemples de colonnes textuelles riches (n>50 uniq): date

- Exemples de prédicteurs numériques (échantillon):
  - price
  - bedrooms
  - bathrooms
  - sqft_living
  - sqft_lot
  - floors
  - waterfront
  - view
  - condition
  - grade

- Missing% on top predictors (sample):
  - price: 0 (0.00%)
  - bedrooms: 0 (0.00%)
  - bathrooms: 0 (0.00%)
  - sqft_living: 0 (0.00%)
  - sqft_lot: 0 (0.00%)
  - floors: 0 (0.00%)
  - waterfront: 0 (0.00%)
  - view: 0 (0.00%)
  - condition: 0 (0.00%)
  - grade: 0 (0.00%)

- Top corr. absolues avec la cible:
  - sqft_living: 0.702
  - grade: 0.667
  - sqft_above: 0.606
  - sqft_living15: 0.585
  - bathrooms: 0.525
  - view: 0.397
  - sqft_basement: 0.324
  - bedrooms: 0.308
  - lat: 0.307
  - waterfront: 0.266

- Top mutual info:
  - zipcode: 0.4278
  - sqft_living: 0.3525
  - lat: 0.3419
  - grade: 0.3360
  - sqft_living15: 0.2700
  - sqft_above: 0.2612
  - bathrooms: 0.2071
  - long: 0.1127
  - id: 0.1089
  - sqft_lot15: 0.0822

---

## Recommandation automatique

- Score houses_Madrid.csv: 7.80
- Score kc_house_data.csv: 9.00
**Choix recommandé**: `kc_house_data.csv`


_Notes_: cette recommandation est heuristique — examinez les corrélations, la présence d'une colonne cible, et la qualité des features avant décision finale._
