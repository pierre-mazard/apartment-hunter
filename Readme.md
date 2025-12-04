
# Apartment Hunter — Guide rapide

Outil prototype pour estimer les prix immobiliers et évaluer la qualité de jeux de données.

Objectif : fournir un workflow reproductible (exploration, sélection de features, modélisation, API).

Contenu principal du dépôt
- `apps/` : applications interactives (Streamlit).
- `scripts/` : scripts d'analyse et de synthèse (génération de rapports Markdown).
- `data/` : jeux de données d'exemple.
- `docs/` : rapports et livrables générés.
- `notebooks/` : notebooks d'exploration et de modélisation.

Prérequis
- Python 3.9+ (utiliser un virtualenv)
- Installer dépendances :

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Exemples d'utilisation
- Générer un rapport d'informativeness :

```bash
python scripts/assess_informativeness.py
```

- Comparaison rapide et génération d'un Markdown :

```bash
python scripts/compare_datasets.py
```

- Lancer l'application Streamlit (exploration interactive) :

```bash
streamlit run apps/compare_datasets_streamlit.py
```

Conseils rapides
- Activez le virtualenv avant d'installer les dépendances.
- Pour afficher la régression dans l'app Streamlit, installez `statsmodels`.

---

