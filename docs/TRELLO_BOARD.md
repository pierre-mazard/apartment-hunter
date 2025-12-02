# Board Trello — Apartment Hunter (Maquette)

Ce fichier contient la structure recommandée du board Trello. 

## Listes
- Backlog
- To Do
- In Progress
- Review / QA
- Done

---

## Cartes (par liste)

### Backlog
- Collecte dataset A / dataset B
  - Estimation: 0.5j
  - Checklist:
    - [ ] Télécharger les fichiers
    - [ ] Vérifier la licence / anonymisation
- Préparer environnement (venv, requirements)
  - Estimation: 0.25j
  - Checklist:
    - [ ] Créer `.venv`
    - [ ] Installer `requirements.txt`
- Définir la feuille de route et les rôles
  - Estimation: 0.5j

### To Do
- Exploration initiale des données (`notebooks/01_exploration.ipynb`)
  - Estimation: 2j
  - Checklist:
    - [ ] Chargement dataset
    - [ ] Statistiques descriptives
    - [ ] Valeurs manquantes
    - [ ] Duplicates
    - [ ] Outliers
- Feature engineering & sélection
  - Estimation: 2j
  - Checklist:
    - [ ] Création features (prix/m2, age, pièces, etc.)
    - [ ] Encodage catégories
    - [ ] Sélection via Boruta/RFE/Lasso
- Entraînement : Linear Regression / RandomForest / XGBoost
  - Estimation: 3j
  - Checklist:
    - [ ] Split train/test
    - [ ] CV et métriques (RMSE, MAE, R2)

### In Progress
- Tuning & GridSearch / RandomizedSearch
  - Estimation: 1-2j
  - Checklist:
    - [ ] Définir grilles
    - [ ] Lancer recherche
    - [ ] Sauvegarder meilleurs modèles
- API Flask `/predict` (intégration du modèle)
  - Estimation: 1j
  - Checklist:
    - [ ] Charger `models/model.pkl`
    - [ ] Valider entrée JSON
    - [ ] Gérer erreurs

### Review / QA
- Tests unitaires pour endpoint `/predict`
  - Estimation: 0.5j
  - Checklist:
    - [ ] Cas succès
    - [ ] Cas input invalide
- Relecture Notebook (clean & comments)
  - Estimation: 0.5j

### Done
- README finalisé
- Dashboard Power BI exporté

---

## Étiquettes (suggestions)
- Urgent
- Haute
- Moyenne
- Basse

## Conseils rapides pour création sur Trello
1. Créez un nouveau board nommé `apartment-hunter - maquette`.
2. Ajoutez les listes dans l'ordre indiquées (Backlog → Done).
3. Pour chaque carte ci‑dessus, créez une carte et collez la checklist.
4. Ajoutez des membres et des dates d'échéance aux cartes prioritaires.
5. Utilisez les étiquettes pour prioriser et les commentaires pour l'avancement.


