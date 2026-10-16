import pandas as pd

from src.features import engineer_features
from src.preprocessing import clean_energy_data


def test_clean_energy_data_sorts_deduplicates_and_fills_missing_values():
    timestamp = pd.Timestamp("2025-01-01")
    data = pd.DataFrame(
        {
            "timestamp": [timestamp + pd.Timedelta(hours=1), timestamp, timestamp],
            "energy_kwh": ["invalid", 10, 10],
            "temperature_c": [22, 20, 20],
            "occupancy_score": [None, 50, 50],
        }
    )

    cleaned = clean_energy_data(data)

    assert len(cleaned) == 2
    assert cleaned["timestamp"].is_monotonic_increasing
    assert cleaned["energy_kwh"].tolist() == [10, 10]
    assert cleaned["occupancy_score"].tolist() == [50, 50]


def test_engineer_features_creates_calendar_lag_and_rolling_features():
    timestamps = pd.date_range("2025-01-01", periods=80, freq="h")
    data = pd.DataFrame(
        {
            "timestamp": timestamps,
            "energy_kwh": range(80),
            "temperature_c": [20] * 80,
            "occupancy_score": [50] * 80,
        }
    )

    featured = engineer_features(data)

    assert featured.loc[12, "hour"] == 12
    assert featured.loc[12, "is_peak_hour"] == 1
    assert featured.loc[72, "day_of_week"] == 5
    assert featured.loc[72, "is_weekend"] == 1
    assert featured.loc[30, "lag_1"] == 29
    assert featured.loc[30, "rolling_3h"] == 28
