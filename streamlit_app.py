from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.arima_model import forecast_sarima, train_sarima_model
from src.data_generator import generate_synthetic_energy_data
from src.data_loader import load_energy_data
from src.features import engineer_features
from src.modeling import train_and_evaluate_model
from src.preprocessing import clean_energy_data


@st.cache_data
def load_or_create_data():
    root = Path(__file__).resolve().parent
    data_dir = root / "data"
    data_dir.mkdir(exist_ok=True)
    csv_path = data_dir / "synthetic_energy_data.csv"

    if not csv_path.exists():
        df = generate_synthetic_energy_data()
        df.to_csv(csv_path, index=False)
    else:
        df = pd.read_csv(csv_path)
    return df


@st.cache_resource
def train_rf_model():
    df = load_or_create_data()
    df = load_energy_data(Path(__file__).resolve().parent / "data" / "synthetic_energy_data.csv")
    df = clean_energy_data(df)
    df = engineer_features(df)
    metrics, model, _, y_test, y_pred = train_and_evaluate_model(df)
    return df, metrics, model, y_test, y_pred


@st.cache_resource
def train_sarima_forecast(df, horizon):
    model_result = train_sarima_model(df)
    forecast = forecast_sarima(model_result, steps=horizon)
    return model_result, forecast


def render_rf_section(metrics, model, y_test, y_pred):
    st.subheader("Predictive Performance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model", "Random Forest")
    col2.metric("MAE", f"{metrics['mae']:.2f}")
    col3.metric("RMSE", f"{metrics['rmse']:.2f}")
    col4.metric("R²", f"{metrics['r2']:.4f}")

    st.subheader("Actual vs Predicted Energy Usage")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_test.values[:200], label="Actual", color="royalblue")
    ax.plot(y_pred[:200], label="Predicted", color="firebrick", linestyle="--")
    ax.set_title("Energy Consumption Forecast")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Energy (kWh)")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Feature Importance")
    feature_names = model.feature_names_in_
    importances = model.feature_importances_
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    importance_df = importance_df.sort_values("Importance", ascending=False)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.barh(importance_df["Feature"], importance_df["Importance"])
    ax2.set_title("Model Feature Importance")
    ax2.set_xlabel("Importance")
    ax2.invert_yaxis()
    st.pyplot(fig2)


def render_sarima_section(filtered_df, forecast_horizon):
    _, forecast = train_sarima_forecast(filtered_df, forecast_horizon)
    last_timestamp = filtered_df["timestamp"].max()
    future_dates = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1), periods=forecast_horizon, freq="h")
    forecast_df = pd.DataFrame({"timestamp": future_dates, "forecast_energy_kwh": forecast["energy_kwh"].values})

    st.subheader("Forecast Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model", "SARIMA")
    col2.metric("Window", f"{forecast_horizon} hrs")
    col3.metric("Mean Forecast", f"{forecast_df['forecast_energy_kwh'].mean():.2f} kWh")
    col4.metric("Last Forecast", f"{forecast_df['forecast_energy_kwh'].iloc[-1]:.2f} kWh")

    st.subheader("Future Energy Forecast")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(forecast_df["timestamp"], forecast_df["forecast_energy_kwh"], color="darkgreen", linewidth=2)
    ax.set_title("SARIMA Energy Forecast")
    ax.set_xlabel("Time")
    ax.set_ylabel("Forecasted Energy (kWh)")
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)

    st.subheader("Forecast Table")
    st.dataframe(forecast_df.head(12), use_container_width=True)


def main():
    st.set_page_config(page_title="Energy Forecasting Dashboard", layout="wide")
    st.title("AI-Powered Energy Consumption Forecasting")
    st.caption("Synthetic smart-grid demand forecasting demo for energy optimization and planning.")

    df = load_or_create_data()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    st.sidebar.header("Controls")
    model_choice = st.sidebar.selectbox("Forecast model", ["Random Forest", "SARIMA"])
    start_date = st.sidebar.date_input("Start date", value=df["timestamp"].min().date())
    end_date = st.sidebar.date_input("End date", value=df["timestamp"].max().date())
    forecast_horizon = st.sidebar.slider("Forecast horizon (hours)", 24, 168, 48)

    if start_date > end_date:
        st.error("Start date must be on or before the end date.")
        st.stop()

    start_timestamp = pd.Timestamp(start_date)
    end_timestamp = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    filtered_df = df[
        (df["timestamp"] >= start_timestamp)
        & (df["timestamp"] < end_timestamp)
    ]
    if filtered_df.empty:
        st.error("No energy data is available for the selected date range.")
        st.stop()

    summary_cols = st.columns(4)
    summary_cols[0].metric("Dataset Rows", len(df))
    summary_cols[1].metric("Selected Window", f"{len(filtered_df)} hrs")
    summary_cols[2].metric("Peak Demand", f"{df['energy_kwh'].max():.1f} kWh")
    summary_cols[3].metric("Avg Demand", f"{df['energy_kwh'].mean():.1f} kWh")

    st.markdown("### Business Summary")
    st.markdown(
        "This project simulates how energy and utility teams forecast demand to reduce waste, balance supply, and lower operational costs."
    )

    if model_choice == "Random Forest":
        _, metrics, model, y_test, y_pred = train_rf_model()
        render_rf_section(metrics, model, y_test, y_pred)
    else:
        render_sarima_section(filtered_df, forecast_horizon)

    st.subheader("Sample Data Preview")
    st.dataframe(filtered_df.head(10), use_container_width=True)

    st.markdown("### Business Insight")
    st.markdown(
        "Use this platform to compare predicted demand against expected operation cycles, identify peak periods, and support planning for smart cities, industrial facilities, and sustainable energy management."
    )


if __name__ == "__main__":
    main()
