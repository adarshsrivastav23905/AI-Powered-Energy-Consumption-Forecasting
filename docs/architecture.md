# Architecture Overview

## System Flow

1. Synthetic energy data is generated or loaded from the `data/` folder.
2. The data is cleaned and sorted by timestamp.
3. Time-based and lag-based features are engineered.
4. Model training uses a RandomForestRegressor.
5. The trained model predicts future or unseen energy demand.
6. Metrics and plots are saved in `outputs/`.

## Key Modules

- `src/data_generator.py`: creates realistic synthetic hourly energy data.
- `src/data_loader.py`: reads the dataset.
- `src/preprocessing.py`: removes invalid values and sets data quality.
- `src/features.py`: adds hourly, weekly, lag, and rolling features.
- `src/modeling.py`: trains the forecasting model.
- `src/visualization.py`: saves comparison charts and feature importance.

## Business View

The model mimics how real utilities, smart-city systems, and building managers make demand forecasts. It supports energy planning, cost control, and better distribution decisions.
