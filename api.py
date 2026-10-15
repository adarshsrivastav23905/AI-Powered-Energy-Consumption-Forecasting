from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request


MODEL_PATH = Path(__file__).resolve().parent / "models" / "energy_api_mlp.pkl"


def create_app(model=None):
    app = Flask(__name__)

    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"API model not found at {MODEL_PATH}. Run `python train_api_model.py` first."
            )
        model = joblib.load(MODEL_PATH)

    @app.post("/predict")
    def predict():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Request body must be a JSON object."}), 400

        hour = payload.get("hour")
        day = payload.get("day")
        if isinstance(hour, bool) or not isinstance(hour, int) or not 0 <= hour <= 23:
            return jsonify({"error": "`hour` must be an integer from 0 to 23."}), 400
        if isinstance(day, bool) or not isinstance(day, int) or not 0 <= day <= 6:
            return jsonify({"error": "`day` must be an integer from 0 (Monday) to 6 (Sunday)."}), 400

        features = pd.DataFrame([{"hour": hour, "day": day}])
        prediction = float(model.predict(features)[0])
        return jsonify({"predicted_energy_kwh": prediction})

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
