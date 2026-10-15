# Project Interview Q&A

## Q1. What was the objective of this project?

The project demonstrates how historical energy measurements can be used to estimate future demand. The goal is to support planning for buildings, utilities, and smart-city systems by making demand patterns easier to anticipate.

## Q2. Why is energy forecasting useful?

Demand forecasts help operators prepare for high-load periods, plan capacity, and assess opportunities to shift flexible consumption. Forecasts are decision-support estimates; they do not by themselves prevent outages or guarantee cost savings.

## Q3. What data did you use?

This repository uses a generated synthetic hourly dataset. It simulates plausible time, weather, occupancy, and demand patterns; it is not corporate smart-grid telemetry or a real utility dataset.

## Q4. Which models did you implement?

The main tabular model is a Random Forest regressor using time, lag, rolling, temperature, and occupancy features. There is also a SARIMA seasonal time-series example. The Flask API demonstrates a separate small MLP regressor that uses only hour and weekday, matching the simple API input contract.

## Q5. How did you train and evaluate the models?

The project reports MAE, RMSE, and R² for its baseline model. The API MLP training script uses a chronological 80/20 split and prints its metrics. The simple API model is a demonstration baseline and does not use the full feature set of the Random Forest model.

## Q6. How is the model served?

The Flask API loads a locally saved MLP artifact created by `python train_api_model.py`. A `POST /predict` request accepts JSON containing `hour` (0–23) and `day` (0–6, Monday–Sunday) and returns an energy estimate in kWh.

## Q7. Is this deployed to a production energy system?

No. The project runs locally and uses simulated data. It demonstrates an end-to-end workflow, not a production utility integration or a live operational forecast.

## Q8. What are the main limitations and next steps?

The simple API model only sees hour and weekday and cannot account for weather, occupancy, recent load, holidays, or unusual events. Next steps include validated real data, time-series backtesting, uncertainty intervals, drift monitoring, and deployment behind production security and reliability controls.

## Claims and evidence

This project avoids unsourced market-size, hiring, salary, national-loss, and percentage-savings claims. Add those to a presentation only when reliable primary sources and dates are available.
