from flask import Flask, request, render_template_string, jsonify
import joblib
import os
from pathlib import Path

from src.config import init_app

app = Flask(__name__)
init_app(app)

# Load model using the configured path
MODEL_PATH = Path(app.config.get('MODEL_PATH'))
model = None
if MODEL_PATH.exists():
    try:
        model = joblib.load(MODEL_PATH)
    except Exception:
        model = None

HOME_HTML = """
<html>
  <head><title>Apartment Hunter</title></head>
  <body>
    <h1>Apartment Hunter — Estimation</h1>
    <p>POST JSON to <code>/predict</code> with features to get a price estimate.</p>
  </body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HOME_HTML)


@app.route('/predict', methods=['POST'])
def predict():
    global model
    if model is None:
        return jsonify({'error': 'Aucun modèle trouvé. Entraînez et sauvegardez un modèle dans /models/model.pkl'}), 500

    data = request.get_json()
    if not data:
        return jsonify({'error': 'JSON body attendu'}), 400

    # Expecting a list or dict of features depending on model
    try:
        # Simple case: single example dict -> convert to list in fixed order
        if isinstance(data, dict):
            X = [list(data.values())]
        else:
            X = data
        pred = model.predict(X)
        return jsonify({'prediction': pred[0] if hasattr(pred, '__len__') else float(pred)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.getenv('PORT', app.config.get('PORT', 5000)))
    app.run(host='0.0.0.0', port=port, debug=app.config.get('DEBUG', True))
