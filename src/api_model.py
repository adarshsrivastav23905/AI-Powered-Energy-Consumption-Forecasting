from pathlib import Path

import joblib
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def train_api_model(df: pd.DataFrame):
    """Train the small hour/day MLP used by the example prediction API."""
    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values("timestamp")

    features = pd.DataFrame(
        {
            "hour": data["timestamp"].dt.hour,
            "day": data["timestamp"].dt.dayofweek,
        }
    )
    target = data["energy_kwh"]

    model = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            max_iter=500,
            early_stopping=True,
            random_state=42,
        ),
    )
    model.fit(features, target)
    return model


def save_api_model(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
