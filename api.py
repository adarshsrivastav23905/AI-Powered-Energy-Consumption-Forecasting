from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request


MODEL_PATH = Path(__file__).resolve().parent / "models" / "energy_api_mlp.pkl"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"API model not found at {MODEL_PATH}. Run `python train_api_model.py` first."
        )
    return joblib.load(MODEL_PATH)


def create_app(model=None):
    app = Flask(__name__)
    model_instance = model

    def get_model():
        nonlocal model_instance
        if model_instance is None:
            model_instance = load_model()
        return model_instance

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
        prediction = float(get_model().predict(features)[0])
        return jsonify({"predicted_energy_kwh": prediction})

    return app


app = create_app()


if __name__ == "__main__":
    app = create_app(model=load_model())
    app.run(host="127.0.0.1", port=5000, debug=False)
