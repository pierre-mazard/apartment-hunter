# Apartment Hunter

Estimation de prix immobiliers (squelette initial)

## Contexte
Développemment d'un outil d'estimation du prix des biens immobiliers via des techniques de Machine Learning. 

## Objectifs du rendu
- Fournir 1-2 notebooks Python documentés (exploration + modélisation).
- Dashboard Power BI exporté.
- README complet expliquant la démarche.
- Script Flask déployant le modèle.
- `Dockerfile` pour conteneuriser l'app.

## Structure du dépôt
```
apartment_hunter/
├─ src/                 # code Flask, utilitaires
├─ notebooks/           # notebooks d'exploration et modélisation
├─ models/              # modèles entraînés (.pkl)
├─ docs/                # roadmap, ressources, livrables
├─ .gitignore
├─ Dockerfile
├─ requirements.txt
└─ README.md
```

## Prérequis
- Python 3.9+
- Docker (optionnel, pour conteneurisation)
- (Optionnel) CLI `gh` pour créer repo GitHub

## Installation locale (dev)
1. Créer et activer un environnement virtuel (utiliser `python3`):
```
python3 -m venv .venv
source .venv/bin/activate
# mettre pip à jour dans le venv puis installer les dépendances
python -m pip install --upgrade pip
pip install -r requirements.txt
```
2. Lancer l'application Flask en développement:
```
# option A — exécution directe (développement)
export FLASK_APP=src/app.py
export FLASK_ENV=development
python -m flask run --host=127.0.0.1 --port=5000

# option B — exécution via le module app (utile si vous avez besoin de la config chargée)
python -m src.app
```

Remarque: si vous obtenez une erreur `externally-managed-environment` lors de l'installation, assurez-vous d'avoir bien activé le virtualenv (`source .venv/bin/activate`) et d'utiliser `python -m pip` depuis ce venv. Sur certaines distributions Debian/Ubuntu, il peut être nécessaire d'installer `python3-venv` (`sudo apt install python3-venv`) avant de créer le venv.

## Utilisation de `.env`
Copiez le fichier d'exemple et personnalisez-le:
```
cp .env.example .env
# éditez .env (SECRET_KEY, MODEL_PATH, PORT si besoin)
```
Les variables du fichier `.env` seront chargées en développement grâce à `python-dotenv`.

## Exécution en production (Docker)
Le `Dockerfile` démarre l'application avec `gunicorn`. Vous pouvez passer des variables d'environnement au conteneur pour configurer le chemin du modèle et le port:
```
docker build -t apartment-hunter:latest .
docker run -e MODEL_PATH=/app/models/model.pkl -e PORT=5000 -p 5000:5000 apartment-hunter:latest
```
Si vous préférez lancer sans Docker, utilisez `gunicorn` (installé via `requirements.txt`):
```
# Exemple local (venv activé)
gunicorn -w 4 -b 0.0.0.0:5000 src.app:app
```

## Docker
Construire et lancer:
```
docker build -t apartment-hunter:latest .
docker run -p 5000:5000 apartment-hunter:latest
```

## Workflow conseillé
- 1) Exploration et nettoyage des données dans `notebooks/01_exploration.ipynb`.
- 2) Sélection de features et entraînement dans `notebooks/02_modeling.ipynb`.
- 3) Sérialiser le modèle final dans `models/model.pkl`.
- 4) Intégrer le chargement du modèle dans `src/app.py` et tester l'endpoint `/predict`.
- 5) Conteneuriser et préparer la présentation PowerPoint/diapositives.

## Livrables attendus
- Notebooks (.ipynb)
- Dashboard Power BI
- README détaillé
- `src/app.py` et `Dockerfile`

---

