import pandas as pd


def engineer_features(df):
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_peak_hour"] = df["hour"].isin([12, 13, 14, 15, 16, 17, 18, 19]).astype(int)

    for lag in [1, 2, 3, 6, 12, 24, 48, 72]:
        df[f"lag_{lag}"] = df["energy_kwh"].shift(lag)

    df["rolling_3h"] = df["energy_kwh"].shift(1).rolling(window=3, min_periods=1).mean()
    df["rolling_24h"] = df["energy_kwh"].shift(1).rolling(window=24, min_periods=1).mean()

    df = df.bfill().ffill()
    return df
