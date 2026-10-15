from __future__ import annotations

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def train_sarima_model(df: pd.DataFrame, target_column: str = "energy_kwh", order=(2, 1, 2), seasonal_order=(1, 0, 1, 24), max_rows: int | None = 2000):
    ts = df[["timestamp", target_column]].copy()
    ts["timestamp"] = pd.to_datetime(ts["timestamp"])
    ts = ts.set_index("timestamp").sort_index()
    if max_rows is not None:
        ts = ts.tail(max_rows)
    ts = ts.asfreq("h")

    model = SARIMAX(
        ts[target_column],
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit(disp=False)
    return result


def forecast_sarima(result, steps: int = 24):
    forecast = result.forecast(steps=steps)
    forecast_df = forecast.reset_index()
    if forecast_df.columns.size == 1:
        forecast_df.columns = ["timestamp", "energy_kwh"]
    elif forecast_df.columns[0] == "index":
        forecast_df.columns = ["timestamp", "energy_kwh"]
    return forecast_df
