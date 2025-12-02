# Feuille de route — Apartment Hunter

## Objectif général
Construire un outil reproductible d'estimation de prix immobiliers, documenté et déployable.

## Phases, tâches et durées estimées
- 1) Collecte des données : 0.5 jour
- 2) Exploration & nettoyage : 2-3 jours
  - gérer valeurs manquantes, duplicates, outliers...
- 3) Feature engineering & sélection : 2 jours
- 4) Modélisation (3 algorithmes + tuning) : 3-4 jours
- 5) Packaging + API Flask : 1-2 jours
- 6) Dockerisation & tests : 0.5-1 jour
- 7) Présentation & documentation (slides + README) : 1-2 jours

## Ressources humaines et matérielles
- 1-3 personnes (idéal pour partager exploration / modelling / dev)
- Machine de dev: CPU standard (4 cores, 8-16GB RAM). GPU non nécessaire pour régressions classiques.

## Critères d'acceptation
- Notebooks propres et commentés.
- API Flask fonctionnelle pour tests locaux.
- Dockerfile permettant exécution sans configuration additionnelle.
