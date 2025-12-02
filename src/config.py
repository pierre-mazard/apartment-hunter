"""Configuration loader.

Utilisez `python-dotenv` en développement : il charge automatiquement les
variables définies dans un fichier `.env` si présent. Le module expose la
classe `Config` que vous pouvez importer dans `src/app.py`.
"""
from __future__ import annotations
import os
from pathlib import Path

try:
    # Charge les variables depuis .env si présent (dev)
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # python-dotenv n'est pas strictement nécessaire en production
    pass


BASE_DIR = Path(__file__).resolve().parent.parent


class Config:
    """Configuration principale — lit depuis les variables d'environnement."""
    SECRET_KEY: str = os.getenv('SECRET_KEY', 'dev-secret')
    FLASK_ENV: str = os.getenv('FLASK_ENV', 'production')
    DEBUG: bool = os.getenv('FLASK_DEBUG', '0') in ('1', 'true', 'True')
    MODEL_PATH: str = os.getenv('MODEL_PATH', str(BASE_DIR / 'models' / 'model.pkl'))
    PORT: int = int(os.getenv('PORT', 5000))


def init_app(app):
    """Applique la configuration à une instance Flask."""
    app.config.from_object(Config)
