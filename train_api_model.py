from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.api_model import save_api_model, train_api_model
from src.data_generator import generate_synthetic_energy_data


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "synthetic_energy_data.csv"
MODEL_PATH = ROOT / "models" / "energy_api_mlp.pkl"


def main():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    else:
        df = generate_synthetic_energy_data()
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_PATH, index=False)

    df = df.sort_values("timestamp").reset_index(drop=True)
    split_index = int(len(df) * 0.8)
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    model = train_api_model(train_df)
    test_features = pd.DataFrame(
        {
            "hour": test_df["timestamp"].dt.hour,
            "day": test_df["timestamp"].dt.dayofweek,
        }
    )
    predictions = model.predict(test_features)
    actual = test_df["energy_kwh"]

    save_api_model(model, MODEL_PATH)
    print(f"Saved hour/day MLP model to {MODEL_PATH}")
    print(f"MAE: {mean_absolute_error(actual, predictions):.2f}")
    print(f"RMSE: {mean_squared_error(actual, predictions) ** 0.5:.2f}")
    print(f"R2 Score: {r2_score(actual, predictions):.4f}")
    print("Note: this simple API model uses hour and day only; it is a demo baseline.")


if __name__ == "__main__":
    main()
