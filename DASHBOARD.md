# JetRail Demand Lab

A Streamlit portfolio dashboard for the historical JetRail forecasting challenge.
The archive ends in September 2014. This is a historical demonstration, with no
live data feed or independent future accuracy guarantee.

## Run in VS Code

Open a PowerShell terminal in the project directory:

```powershell
cd "D:\passenger prediction\passenger-demand-forecasting"
& "C:\Users\mwang\anaconda3\python.exe" -m pip install -r requirements-dashboard.txt
& "C:\Users\mwang\anaconda3\python.exe" -B -m streamlit run app.py
```

On another computer, use the active environment's `python` in place of the full
interpreter path. The same interpreter must install dependencies and launch the
app. Open the local URL printed by Streamlit, normally http://localhost:8501.

## Pages

- **Overview:** archive coverage, observed hourly counts, hourly demand summaries,
  daily hourly averages, and the trailing seven-day mean.
- **Historical Data Explorer:** date filtering, individual hourly counts,
  daily/monthly sums or hourly averages, calendar averages, and distributions.
  Totals use available observations; missing hours are not imputed.
- **Forecasting:** select a valid development origin, then explicitly generate
  24 hours with the existing `fit_delta_model()` and `recursive_forecast()`.
  Reveal future actuals separately, inspect training chronology, and download CSV.
- **Model Performance:** discover compatible completed `forecast_validation_*`
  experiments; select a run and inspect pooled scores, origin win rates,
  horizon errors/bias, origin stability, and actual-demand strata.
- **Model Insights:** explanations of features, the delta target, recursion,
  walk-forward validation, and existing saved distribution-shift diagnostics.

## Data and integrity

The overview/explorer read `data/train.csv` using the existing raw-data validator.
Forecasting reads and audits `data/passenger_features.csv` using the established
development loader. All model inputs and forecast targets are restricted to the
original pre-test development period. Eligible origins require 180 days of
history, complete 168-hour context, and 24 subsequent observed development hours.

Labels strictly precede the origin. Context includes its observed count. Later
features are recomputed from predicted counts, and there is no within-horizon
refitting. The earlier absolute-demand joblib model is never loaded. No model or
experiment artifact is written by the dashboard.

Forecast cache keys include the origin, dataset SHA256, imported forecasting
source hashes, ordered features, RF parameters, and numerical library versions.
Changing the selected origin does not train a model; the Generate button requests
computation. Revealing actual outcomes uses the completed forecast and does not
refit. In-memory forecast caching is bounded to 12 entries.

Experiment discovery reads `validation_settings.json`, requires `status=complete`,
and checks method compatibility, including imported forecasting source hashes.
It verifies saved forecast counts, chronology,
strategy alignment, errors, demand thresholds, and metric tables before display.
The latest run is determined by the metadata file's modification time because
the current metadata schema has no explicit completion timestamp. Copying files
can change that order; use the run selector to choose a particular experiment.

Weekly origins sample a fixed weekday composition. Daily windows abut, expanding
training sets overlap, and demand is serially correlated. Origin scores are not
independent replicates. Lower historical errors do not establish future
generalization. New unseen chronological data are required for independent
evaluation. Existing inspected test statistics appear only as retrospective
descriptive information in the explorer/insights.

## Manual acceptance checks

1. Open all five pages. Verify missing-file messages are actionable rather than
   tracebacks. The app can show insights/performance without loading raw data.
2. In the explorer, select a short range. Compare daily totals with the sum of
   displayed hourly observations and verify the average option changes units.
3. On Forecasting, generate an origin. Confirm exactly 24 unique timestamps,
   from origin + 1 hour through origin + 24 hours, with nonnegative predictions.
4. Expand the training audit: training ends strictly before the origin, and
   feature order and RF settings match the existing forecasting scripts.
5. Toggle actual outcomes. Predictions must remain unchanged. Download the CSV;
   it contains predictions and audit columns, never future actuals.
6. Select another origin: the previous forecast is hidden until Generate is
   pressed. Returning to the same origin uses the cache if data/code are unchanged.
7. On Model Performance, select the 67-origin weekly experiment. Confirm 1,608
   forecast cases per strategy and inspect both RF wins and losses. Select the
   daily experiment for broader weekday coverage; do not compare different
   samples as if they isolate model improvement.

For automated UI testing, Streamlit's built-in `streamlit.testing.v1.AppTest`
can execute the app and manipulate the navigation and widgets. Forecast validity
can be checked against the imported `forecast_24h` functions on the same origin.
