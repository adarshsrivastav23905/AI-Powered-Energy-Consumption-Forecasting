import numpy as np
import pandas as pd


def generate_synthetic_energy_data():
    start = pd.Timestamp("2023-01-01 00:00:00")
    end = pd.Timestamp("2025-12-31 23:00:00")
    timestamps = pd.date_range(start=start, end=end, freq="h")

    df = pd.DataFrame({"timestamp": timestamps})
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["day_of_year"] = df["timestamp"].dt.dayofyear

    temp_trend = 18 + 12 * np.sin((df["day_of_year"] / 365) * 2 * np.pi - 1.5)
    df["temperature_c"] = temp_trend + np.random.normal(0, 3, len(df))

    occupancy_base = 40 + 30 * np.sin((df["hour"] / 24) * 2 * np.pi)
    occupancy_base += 15 * (df["day_of_week"] >= 5)
    df["occupancy_score"] = (occupancy_base + np.random.normal(0, 5, len(df))).clip(0, 100)

    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_peak_hour"] = df["hour"].isin([12, 13, 14, 15, 16, 17, 18, 19]).astype(int)
    df["holiday_flag"] = ((df["month"] == 12) & (df["day_of_year"] > 350)).astype(int)

    daily_cycle = 90 * np.sin((df["hour"] / 24) * 2 * np.pi - 1.3)
    weekly_cycle = 35 * np.sin((df["day_of_week"] / 7) * 2 * np.pi)
    temp_effect = 3 * (df["temperature_c"] - 20)
    occupancy_effect = 0.8 * df["occupancy_score"]
    peak_effect = 40 * df["is_peak_hour"]
    weekend_effect = 15 * df["is_weekend"]
    holiday_effect = 25 * df["holiday_flag"]
    noise = np.random.normal(0, 8, len(df))

    df["energy_kwh"] = (
        120
        + daily_cycle
        + weekly_cycle
        + temp_effect
        + occupancy_effect
        + peak_effect
        + weekend_effect
        + holiday_effect
        + noise
    )

    df["energy_kwh"] = df["energy_kwh"].clip(lower=0)
    return df
