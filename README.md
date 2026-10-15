# AI-Powered Energy Consumption Forecasting System

A practical machine learning project for forecasting electricity usage in homes, buildings, campuses, and small industrial systems using synthetic but realistic energy data.

## Project Overview

This project simulates a real-world smart-grid and energy optimization use case. It helps answer a common operational question in energy management:

> How much electricity will be consumed in the next hour, day, or week?

By forecasting demand, cities, buildings, and power operations can reduce waste, manage peak loads, lower energy bills, and improve sustainability.

## Why This Project Matters

Modern electricity systems need to balance supply and demand continuously. If demand is underestimated, there may be outages or instability. If demand is overestimated, utilities and organizations may waste generation capacity and money.

This project addresses that through a data-driven approach.

## Problem Statement

Energy demand is often difficult to predict because it changes with:
- time of day
- day of week
- weather conditions
- occupancy and usage behavior
- seasonal patterns
- industrial or commercial activity

Without forecasting, organizations struggle with:
- unexpected electricity spikes
- excessive power cost during peak periods
- inefficient operation of HVAC and equipment
- unstable grid planning
- poor renewable energy scheduling

## Business Value

The system helps organizations:
- plan supply and demand more effectively
- reduce peak-time electricity costs
- identify wasteful consumption patterns
- improve smart-city energy management
- support sustainability goals and carbon reduction

## Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- RandomForestRegressor
- Streamlit
- Flask
- Jupyter Notebook

## Project Architecture

```text
Raw energy data
      |
      v
Data cleaning and validation
      |
      v
Feature engineering (hour, weekday, lag, rolling, occupancy)
      |
      v
Train-test split
      |
      v
Random Forest regression model
      |
      v
Prediction and evaluation
      |
      v
Visualization and business insight
```

## Folder Structure

```text
AI-Energy-Forecasting/
├── data/
│   └── synthetic_energy_data.csv
├── docs/
│   └── architecture.md
├── images/
├── models/
│   └── random_forest_energy_model.pkl
├── notebooks/
│   └── energy_forecasting_demo.ipynb
├── outputs/
│   ├── actual_vs_predicted.png
│   └── feature_importance.png
├── src/
│   ├── __init__.py
│   ├── api_model.py
│   ├── data_generator.py
│   ├── data_loader.py
│   ├── features.py
│   ├── modeling.py
│   ├── preprocessing.py
│   └── visualization.py
├── .gitignore
├── main.py
├── api.py
├── train_api_model.py
├── README.md
├── requirements.txt
├── streamlit_app.py
└── .venv/
```

## Features Included

- Synthetic energy dataset generation
- Hourly time-series data simulation
- Data cleaning and preprocessing
- Feature engineering with lag and rolling features
- Machine learning forecast model
- SARIMA time-series forecasting option for seasonal demand trends
- Model evaluation with MAE, RMSE, and R²
- Plot generation for actual vs predicted energy usage
- Streamlit dashboard with time filters and forecast horizon controls
- Flask prediction API with validated JSON inputs
- Separately trained MLP baseline for the API's hour/day prediction contract
- Notebook version for educational use
- GitHub-ready demo asset generation script for screenshots and visuals

## Installation

### 1. Create a virtual environment

Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

Mac/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the project
```bash
python main.py
```

### 4. Launch the dashboard
```bash
streamlit run streamlit_app.py
```

### Flask prediction API

The API uses a separate, small MLP model with `hour` and weekday (`day`) as inputs. Train and save that model first:

```bash
python train_api_model.py
python api.py
```

On Windows PowerShell, send a request to `http://127.0.0.1:5000/predict`:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:5000/predict -Method Post `
  -ContentType "application/json" -Body '{"hour":14,"day":2}'
```

On Mac/Linux, use curl:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"hour":14,"day":2}'
```

The API validates both inputs and returns `predicted_energy_kwh`. It is a local demo API, not a production service.

Project interview preparation is in [docs/interview_qa.md](docs/interview_qa.md).

The serialized API model is generated locally and excluded from Git. Run `python train_api_model.py` after cloning to create it.

## Example Result

This project has been validated with the current pipeline and produces strong predictive performance on synthetic smart-grid data.

Example output:
```text
Forecasting system completed successfully.
MAE: 9.50
RMSE: 12.12
R2 Score: 0.9775
```

## Notebook Version

A demo notebook is available in:
- [notebooks/energy_forecasting_demo.ipynb](notebooks/energy_forecasting_demo.ipynb)

This notebook walks through:
- dataset generation
- data preprocessing
- feature engineering
- model training
- plotting results

## Dashboard Version

The dashboard is available in:
- [streamlit_app.py](streamlit_app.py)

It provides a simple UI to view:
- performance metrics
- actual vs predicted trend
- feature importance chart
- historical demand and model forecast views
- date/model/horizon filters with an apply action
- a responsive KPI layout and a coordinated light theme configured in `.streamlit/config.toml`
- scoped premium styling for the hero, metric cards, filter panel, and chart surfaces in `.streamlit/style.css`

## GitHub Portfolio Value

This project is suitable for:
- internships
- job portfolios
- data science showcases
- machine learning and AI discussions
- energy analytics presentations

It is beginner-friendly, but still demonstrates real-world project thinking.

## Future Improvements

Possible next extensions:
- LSTM or GRU deep learning model
- weather-aware demand forecasting
- real CSV dataset integration
- production hardening of the local Flask API
- cloud-hosted dashboard

## Advanced Forecasting Option

This project includes a seasonal forecasting path using SARIMA for time-series trends and a Random Forest baseline for tabular learning. The advanced version is implemented in:
- [src/arima_model.py](src/arima_model.py)

A deep-learning extension is also included as a notebook:
- [notebooks/lstm_energy_forecasting.ipynb](notebooks/lstm_energy_forecasting.ipynb)

For screenshots and portfolio-ready visuals, run:
```bash
python scripts/generate_demo_assets.py
```

This will save files such as:
- `images/dataset_preview.png`
- `images/daily_energy_trend.png`
- `images/energy_distribution.png`

## Portfolio Page

For a final GitHub-facing project summary and presentation layout, see:
- [PORTFOLIO.md](PORTFOLIO.md)

## Summary

This project demonstrates how AI can be used to forecast energy consumption with practical business value. It is a strong student-friendly portfolio project because it combines machine learning, forecasting, analytics, and real-world energy decision-making.

## Project Goal

Forecast electricity usage in homes, buildings, and industries using AI to support smart cities, climate tech, and sustainable energy planning.

## License

This project is intended for learning and portfolio use.
