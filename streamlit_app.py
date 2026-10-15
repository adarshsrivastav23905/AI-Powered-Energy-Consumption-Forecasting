from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.arima_model import forecast_sarima, train_sarima_model
from src.data_generator import generate_synthetic_energy_data
from src.features import engineer_features
from src.modeling import train_and_evaluate_model
from src.preprocessing import clean_energy_data


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "synthetic_energy_data.csv"


@st.cache_data(max_entries=1)
def load_or_create_data():
    DATA_PATH.parent.mkdir(exist_ok=True)
    if not DATA_PATH.exists():
        generate_synthetic_energy_data().to_csv(DATA_PATH, index=False)
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


@st.cache_resource
def train_rf_model():
    df = engineer_features(clean_energy_data(load_or_create_data()))
    metrics, model, X_test, y_test, y_pred = train_and_evaluate_model(df)

    test_results = pd.DataFrame(
        {
            "timestamp": df.loc[X_test.index, "timestamp"].to_numpy(),
            "Actual demand": y_test.to_numpy(),
            "Predicted demand": y_pred,
        }
    ).sort_values("timestamp")
    return metrics, model, test_results


@st.cache_resource
def train_sarima_forecast(df, horizon):
    result = train_sarima_model(df)
    return forecast_sarima(result, steps=horizon)


def render_kpis(df, selected_df, model_choice, forecast_horizon):
    average_demand = selected_df["energy_kwh"].mean()
    peak_demand = selected_df["energy_kwh"].max()
    with st.container(key="overview-kpis"):
        with st.container(horizontal=True):
            st.metric("Average demand", f"{average_demand:,.1f} kWh", border=True)
            st.metric("Peak demand", f"{peak_demand:,.1f} kWh", border=True)
            st.metric("Observations", f"{len(selected_df):,}", border=True)
            st.metric(
                "Forecast horizon",
                f"{forecast_horizon} hours",
                border=True,
            )


def render_historical_trend(selected_df):
    daily = (
        selected_df.assign(day=selected_df["timestamp"].dt.floor("D"))
        .groupby("day", as_index=False)["energy_kwh"]
        .mean()
        .rename(columns={"day": "Date", "energy_kwh": "Average demand"})
    )
    with st.container(border=True, key="demand-trend-card"):
        st.subheader("Demand trend")
        st.caption("Daily average electricity consumption across the selected period.")
        chart = (
            alt.Chart(daily)
            .mark_line(color="#0E7490", point=alt.OverlayMarkDef(size=35))
            .encode(
                x=alt.X("Date:T", title=None, axis=alt.Axis(format="%b %d", labelAngle=0)),
                y=alt.Y(
                    "Average demand:Q",
                    title="Energy (kWh)",
                    scale=alt.Scale(zero=False),
                ),
                tooltip=[
                    alt.Tooltip("Date:T", title="Date"),
                    alt.Tooltip("Average demand:Q", title="Average demand", format=".1f"),
                ],
            )
            .properties(height=300)
        )
        st.altair_chart(chart, width="stretch")


def render_random_forest(metrics, model, test_results, start_timestamp, end_timestamp):
    with st.container(key="evaluation-kpis"):
        with st.container(horizontal=True):
            st.metric("Mean absolute error", f"{metrics['mae']:.2f} kWh", border=True)
            st.metric("Root mean squared error", f"{metrics['rmse']:.2f} kWh", border=True)
            st.metric("R² score", f"{metrics['r2']:.4f}", border=True)
    st.caption("Holdout evaluation metrics are calculated across the complete dataset.")

    selected_results = test_results[
        (test_results["timestamp"] >= start_timestamp)
        & (test_results["timestamp"] < end_timestamp)
    ]
    if selected_results.empty:
        st.info("No holdout observations fall within this date range. Expand the range to see the comparison chart.")
    else:
        with st.container(border=True, key="actual-predicted-card"):
            st.subheader("Actual vs predicted")
            st.caption("Random Forest holdout observations within the selected date range.")
            chart_data = selected_results.melt(
                id_vars="timestamp",
                var_name="Series",
                value_name="Demand",
            )
            chart = (
                alt.Chart(chart_data)
                .mark_line(strokeWidth=2)
                .encode(
                    x=alt.X(
                        "timestamp:T",
                        title=None,
                        axis=alt.Axis(format="%b %d", labelAngle=0),
                    ),
                    y=alt.Y(
                        "Demand:Q",
                        title="Energy (kWh)",
                        scale=alt.Scale(zero=False),
                    ),
                    color=alt.Color(
                        "Series:N",
                        title=None,
                        scale=alt.Scale(
                            domain=["Actual demand", "Predicted demand"],
                            range=["#0E7490", "#7C3AED"],
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip("timestamp:T", title="Timestamp"),
                        alt.Tooltip("Series:N", title="Series"),
                        alt.Tooltip("Demand:Q", title="Energy (kWh)", format=".1f"),
                    ],
                )
                .properties(height=340)
            )
            st.altair_chart(chart, width="stretch")

    importance = pd.DataFrame(
        {
            "Feature": model.feature_names_in_,
            "Importance": model.feature_importances_,
        }
    ).sort_values("Importance", ascending=True)
    with st.container(border=True, key="feature-importance-card"):
        st.subheader("What drives demand")
        st.caption("Relative feature importance in the Random Forest model.")
        chart = (
            alt.Chart(importance)
            .mark_bar(color="#0E7490", cornerRadiusEnd=4)
            .encode(
                x=alt.X("Importance:Q", title="Relative importance"),
                y=alt.Y("Feature:N", sort="-x", title=None),
                tooltip=[
                    alt.Tooltip("Feature:N", title="Feature"),
                    alt.Tooltip("Importance:Q", title="Importance", format=".3f"),
                ],
            )
            .properties(height=420)
        )
        st.altair_chart(chart, width="stretch")


def render_sarima(selected_df, forecast_horizon):
    with st.spinner("Fitting seasonal model and preparing forecast..."):
        forecast = train_sarima_forecast(selected_df, forecast_horizon)

    last_timestamp = selected_df["timestamp"].max()
    future_timestamps = pd.date_range(
        start=last_timestamp + pd.Timedelta(hours=1),
        periods=forecast_horizon,
        freq="h",
    )
    forecast_df = pd.DataFrame(
        {
            "timestamp": future_timestamps,
            "Forecast demand": forecast["energy_kwh"].to_numpy(),
        }
    )
    forecast_average = forecast_df["Forecast demand"].mean()
    forecast_peak = forecast_df["Forecast demand"].max()
    with st.container(key="forecast-kpis"):
        with st.container(horizontal=True):
            st.metric("Forecast average", f"{forecast_average:,.1f} kWh", border=True)
            st.metric("Forecast peak", f"{forecast_peak:,.1f} kWh", border=True)
            st.metric("Forecast points", f"{len(forecast_df):,}", border=True)

    recent_history = selected_df.tail(min(len(selected_df), 7 * 24))[
        ["timestamp", "energy_kwh"]
    ].rename(columns={"energy_kwh": "Demand"})
    recent_history["Series"] = "Historical demand"
    forecast_plot = forecast_df.rename(columns={"Forecast demand": "Demand"})
    forecast_plot["Series"] = "Forecast"
    combined = pd.concat(
        [
            recent_history,
            forecast_plot,
        ],
        ignore_index=True,
    )
    with st.container(border=True, key="seasonal-forecast-card"):
        st.subheader("Seasonal demand forecast")
        st.caption("Recent history followed by the requested SARIMA forecast horizon.")
        chart = (
            alt.Chart(combined)
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X(
                    "timestamp:T",
                    title=None,
                    axis=alt.Axis(format="%b %d, %H:%M", labelAngle=0),
                ),
                y=alt.Y(
                    "Demand:Q",
                    title="Energy (kWh)",
                    scale=alt.Scale(zero=False),
                ),
                color=alt.Color(
                    "Series:N",
                    title=None,
                    scale=alt.Scale(
                        domain=["Historical demand", "Forecast"],
                        range=["#0E7490", "#F59E0B"],
                    ),
                ),
                tooltip=[
                    alt.Tooltip("timestamp:T", title="Time"),
                    alt.Tooltip("Series:N", title="Series"),
                    alt.Tooltip("Demand:Q", title="Energy (kWh)", format=".1f"),
                ],
            )
            .properties(height=340)
        )
        st.altair_chart(chart, width="stretch")

    with st.container(border=True, key="forecast-schedule-card"):
        st.subheader("Forecast schedule")
        st.dataframe(
            forecast_df.rename(
                columns={
                    "timestamp": "Forecast time",
                    "Forecast demand": "Forecast demand (kWh)",
                }
            ),
            hide_index=True,
            column_config={
                "Forecast time": st.column_config.DatetimeColumn(
                    "Forecast time", format="MMM D, YYYY, h A"
                ),
                "Forecast demand (kWh)": st.column_config.NumberColumn(
                    "Forecast demand (kWh)", format="%.2f"
                ),
            },
        )


def main():
    st.set_page_config(
        page_title="Energy demand | Forecasting",
        page_icon=":material/bolt:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.html(ROOT / ".streamlit" / "style.css")

    df = load_or_create_data()
    minimum_date = df["timestamp"].min().date()
    maximum_date = df["timestamp"].max().date()
    default_start = max(minimum_date, maximum_date - pd.Timedelta(days=29))

    with st.sidebar:
        st.markdown("## :material/bolt: GridSight")
        st.caption("Energy demand intelligence")
        st.badge("Synthetic demo", icon=":material/science:", color="blue")
        st.space("small")
        st.subheader("Forecast controls")
        with st.form(key="filter-form"):
            model_choice = st.selectbox(
                "Forecast model",
                ["Random Forest", "SARIMA"],
                help="Random Forest evaluates a holdout set. SARIMA forecasts future hourly demand.",
            )
            start_date = st.date_input(
                "Start date",
                value=default_start,
                min_value=minimum_date,
                max_value=maximum_date,
            )
            end_date = st.date_input(
                "End date",
                value=maximum_date,
                min_value=minimum_date,
                max_value=maximum_date,
            )
            forecast_horizon = st.slider(
                "Forecast horizon (hours)",
                min_value=24,
                max_value=168,
                value=48,
                step=24,
            )
            apply_filters = st.form_submit_button(
                "Apply settings",
                type="primary",
                icon=":material/tune:",
                width="stretch",
            )

        st.caption("Filters update when you apply settings.")
        st.caption("All demand values are generated synthetic data.")

    if "dashboard_filters" not in st.session_state:
        st.session_state["dashboard_filters"] = {
            "model": model_choice,
            "start": start_date,
            "end": end_date,
            "horizon": forecast_horizon,
        }
    if apply_filters:
        st.session_state["dashboard_filters"] = {
            "model": model_choice,
            "start": start_date,
            "end": end_date,
            "horizon": forecast_horizon,
        }

    filters = st.session_state["dashboard_filters"]
    start_date = filters["start"]
    end_date = filters["end"]
    model_choice = filters["model"]
    forecast_horizon = filters["horizon"]

    if start_date > end_date:
        st.error("Start date must be on or before the end date.", icon=":material/error:")
        st.stop()

    start_timestamp = pd.Timestamp(start_date)
    end_timestamp = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    selected_df = df[
        (df["timestamp"] >= start_timestamp)
        & (df["timestamp"] < end_timestamp)
    ]
    if selected_df.empty:
        st.error("No data exists in the selected date range.", icon=":material/error:")
        st.stop()

    with st.container(border=True, key="hero-card"):
        st.badge("ENERGY ANALYTICS  /  DEMAND PLANNING", color="blue")
        st.title("Energy demand")
        st.markdown("Forecast demand. Plan with confidence.")
        st.caption(
            f"{start_date:%b %d, %Y} — {end_date:%b %d, %Y}  ·  {model_choice}"
        )

    st.subheader("At a glance")
    render_kpis(df, selected_df, model_choice, forecast_horizon)

    left, right = st.columns([1.5, 1], gap="large")
    with left:
        render_historical_trend(selected_df)
    with right:
        selected_summary = pd.DataFrame(
            {
                "Measure": ["Average", "Median", "Peak", "Lowest"],
                "Demand (kWh)": [
                    selected_df["energy_kwh"].mean(),
                    selected_df["energy_kwh"].median(),
                    selected_df["energy_kwh"].max(),
                    selected_df["energy_kwh"].min(),
                ],
            }
        )
        with st.container(border=True, key="period-summary-card"):
            st.subheader("Period summary")
            st.caption("Consumption statistics for the selected dates.")
            st.dataframe(
                selected_summary,
                hide_index=True,
                column_config={
                    "Demand (kWh)": st.column_config.NumberColumn(
                        "Demand (kWh)", format="%.1f"
                    )
                },
            )
            st.caption(f"{len(selected_df):,} hourly observations")

    st.subheader("Forecast analysis")
    if model_choice == "Random Forest":
        with st.spinner("Loading model evaluation..."):
            metrics, model, test_results = train_rf_model()
        render_random_forest(
            metrics, model, test_results, start_timestamp, end_timestamp
        )
    else:
        render_sarima(selected_df, forecast_horizon)

    with st.expander("Explore source data", icon=":material/table_view:"):
        preview = selected_df.head(100).rename(
            columns={
                "timestamp": "Timestamp",
                "energy_kwh": "Energy (kWh)",
                "temperature_c": "Temperature (°C)",
                "occupancy_score": "Occupancy score",
            }
        )
        st.dataframe(
            preview,
            hide_index=True,
            column_config={
                "Timestamp": st.column_config.DatetimeColumn(
                    "Timestamp", format="MMM D, YYYY, h A"
                ),
                "Energy (kWh)": st.column_config.NumberColumn(
                    "Energy (kWh)", format="%.2f"
                ),
                "Temperature (°C)": st.column_config.NumberColumn(
                    "Temperature (°C)", format="%.1f"
                ),
                "Occupancy score": st.column_config.NumberColumn(
                    "Occupancy score", format="%.0f"
                ),
            },
        )
        st.download_button(
            "Download selected data",
            data=selected_df.to_csv(index=False),
            file_name="energy_demand_selected.csv",
            mime="text/csv",
            icon=":material/download:",
        )

    st.caption(
        "Decision-support demo only · Synthetic data · Not for live grid operations"
    )


if __name__ == "__main__":
    main()
