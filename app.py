"""JetRail historical demand dashboard.

Launch with ``python -m streamlit run app.py`` from this directory.
No experiment, fitting, artifact writing, or UI rendering occurs on import.
"""

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sklearn
import streamlit as st

from backtesting import CUTOFF, RF_PARAMS, metrics
from features import FEATURE_COLUMNS, load_dataset
from forecast_24h import (
    BASE, CONTEXT, HOURS, STRATEGIES, choose_origins, fit_delta_model,
    load_development_history, recursive_forecast, validate_recursive_features,
)
from forecast_validation import DEPENDENCE, WEEKDAYS, diagnostic_tables

DATA_PATH = BASE / "data" / "train.csv"
DEVELOPMENT_PATH = BASE / "data" / "passenger_features.csv"
OUTPUT_PATH = BASE / "outputs"
PAGES = ("Overview", "Historical Data Explorer", "Forecasting", "Model Performance", "Model Insights")
COLORS = {STRATEGIES[0]: "#087f8c", STRATEGIES[1]: "#9a6b32", STRATEGIES[2]: "#7586b5"}
TABLE_NAMES = ("overall_summary", "metrics_by_horizon", "metrics_by_origin", "metrics_by_demand",
            "metrics_by_weekday", "metrics_by_day_type", "baseline_comparisons")
ERRORS = (OSError, ValueError, TypeError, KeyError, AssertionError, json.JSONDecodeError)


def fingerprint(path: Path) -> str:
    """Hash actual file contents so dataset/cache identity changes with the data."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path.name}. Expected it in {path.parent}.")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def method_version() -> str:
    """Include imported forecasting code, feature order, parameters and library versions."""
    sources = {name: fingerprint(BASE / name) for name in ("features.py", "backtesting.py", "forecast_24h.py")}
    return json.dumps(dict(sources=sources, features=FEATURE_COLUMNS, params=RF_PARAMS,
                        sklearn=sklearn.__version__, pandas=pd.__version__, numpy=np.__version__), sort_keys=True)


@st.cache_data(show_spinner=False, max_entries=4)
def historical_data(version: str) -> pd.DataFrame:
    """Use the existing raw-data validator; retain gaps as missing observations."""
    if fingerprint(DATA_PATH) != version:
        raise ValueError("Historical data changed while loading. Refresh the page.")
    data = load_dataset(DATA_PATH)
    if fingerprint(DATA_PATH) != version:
        raise ValueError("Historical data changed while loading. Refresh the page.")
    return data


@st.cache_data(show_spinner=False, max_entries=4)
def development_data(version: str) -> pd.DataFrame:
    """Reuse the saved-feature audit and exclusive original-test cutoff."""
    if fingerprint(DEVELOPMENT_PATH) != version:
        raise ValueError("Development data changed while loading. Refresh the page.")
    data = load_development_history()
    if fingerprint(DEVELOPMENT_PATH) != version:
        raise ValueError("Development data changed while loading. Refresh the page.")
    return data


@st.cache_data(show_spinner=False, max_entries=4)
def feature_checks(version: str) -> bool:
    """Run the established recursive parity/isolation checks once per method version."""
    if version != method_version():
        raise ValueError("Forecasting code changed. Refresh before generating forecasts.")
    validate_recursive_features()
    return True


def eligible_origins(history: pd.DataFrame) -> pd.DatetimeIndex:
    """Offer observed hourly origins with 180 days of history and complete outcomes."""
    present = history.Count.notna().astype(int)
    context = present.rolling(CONTEXT, min_periods=CONTEXT).sum().eq(CONTEXT)
    outcomes = present.rolling(HOURS, min_periods=HOURS).sum().shift(-HOURS).eq(HOURS)
    chronological = ((history.index >= history.index.min() + pd.Timedelta(days=180)) &
                    (history.index + pd.Timedelta(hours=HOURS) < CUTOFF))
    return history.index[context & outcomes & chronological]


@st.cache_data(show_spinner=False, max_entries=12)
def generate_forecast(origin_text: str, dataset_version: str, algorithm_version: str,
                    model_settings: str) -> tuple[pd.DataFrame, dict]:
    """Call the exact tested fit/recursion functions; return no future actuals.

    The cache stores only generated tables, never an external serialized model.
    Origin, data bytes, imported code, feature order and RF parameters are keys.
    """
    tick = perf_counter()
    if model_settings != json.dumps(RF_PARAMS, sort_keys=True):
        raise ValueError("Forecast settings differ from the established RF configuration.")
    feature_checks(algorithm_version)
    history = development_data(dataset_version)
    origin = pd.Timestamp(origin_text)
    choose_origins(history, 1, 1, 180, [origin_text])
    observed = history.loc[:origin, ["Count"]].copy()
    model, training = fit_delta_model(observed, origin)
    if list(model.feature_names_in_) != FEATURE_COLUMNS:
        raise ValueError("Fitted feature ordering differs from the feature engineering pipeline.")
    forecast = recursive_forecast(model, observed)
    expected = pd.date_range(origin + pd.Timedelta(hours=1), periods=HOURS, freq="h")
    if (not pd.DatetimeIndex(forecast.Datetime).equals(expected)
            or forecast.Datetime.duplicated().any() or not forecast.Horizon.eq(np.arange(1, HOURS + 1)).all()
            or not np.isfinite(forecast.Predicted_Count).all() or forecast.Predicted_Count.lt(0).any()
            or not forecast.Datetime.lt(CUTOFF).all()):
        raise ValueError("Generated forecast failed timestamp or count validation.")
    forecast.insert(0, "Origin", origin)
    forecast["Strategy"] = STRATEGIES[0]
    training.update(Origin=origin_text, Dataset_SHA256=dataset_version,
                    Generation_Seconds=round(perf_counter() - tick, 2))
    return forecast, training


def check_metadata(settings: dict) -> None:
    """Reject incomplete or incompatible saved strategies before presenting scores."""
    if not isinstance(settings, dict):
        raise ValueError("Experiment metadata must be a JSON object.")
    expected = {"status": "complete", "horizon_hours": HOURS, "target": "Count - lag_1",
                "features": FEATURE_COLUMNS, "rf_params": RF_PARAMS,
                "cutoff_exclusive": str(CUTOFF), "training_labels": "strictly before origin"}
    for key, value in expected.items():
        if settings.get(key) != value:
            raise ValueError(f"Incompatible experiment metadata: {key}.")
    for name in ("features.py", "backtesting.py", "forecast_24h.py"):
        if settings.get("source_sha256", {}).get(name) != fingerprint(BASE / name):
            raise ValueError(f"Experiment uses a different version of {name}.")
    if not isinstance(settings.get("origins_evaluated"), int) or settings["origins_evaluated"] < 1:
        raise ValueError("Experiment has no valid completed origin count.")
    if not isinstance(settings.get("arguments"), dict) or settings["arguments"].get("stride_days") not in (1, 7):
        raise ValueError("Experiment has an unsupported origin stride.")


def experiment_signature(directory: Path) -> str:
    """Fingerprint every CSV and the metadata used to display a completed run."""
    names = ["validation_settings.json", "validation_predictions.csv", "validation_training_windows.csv",
            *[f"validation_{name}.csv" for name in TABLE_NAMES]]
    return json.dumps({name: fingerprint(directory / name) for name in names}, sort_keys=True)


@st.cache_data(show_spinner=False, max_entries=12)
def load_experiment(folder: str, signature: str, compatibility: str) -> tuple[dict, dict[str, pd.DataFrame], pd.DataFrame]:
    """Cross-check completed metadata, timestamps, common outcomes and saved metrics."""
    directory = Path(folder).resolve()
    if directory.parent != OUTPUT_PATH.resolve() or not directory.name.startswith("forecast_validation_"):
        raise ValueError("Experiments must be local forecast_validation output directories.")
    if experiment_signature(directory) != signature or compatibility != method_version():
        raise ValueError("Experiment files or forecasting code changed. Refresh the page.")
    settings = json.loads((directory / "validation_settings.json").read_text(encoding="utf-8"))
    check_metadata(settings)
    forecasts = pd.read_csv(directory / "validation_predictions.csv")
    required = {"Origin", "Datetime", "Horizon", "Strategy", "Actual_Count", "Predicted_Count", "Error", "Absolute_Error"}
    if forecasts.empty or required.difference(forecasts.columns):
        raise ValueError("Saved predictions have an incomplete schema.")
    for column in ("Origin", "Datetime"):
        forecasts[column] = pd.to_datetime(forecasts[column], errors="raise")
        if forecasts[column].dt.tz is not None or forecasts[column].isna().any():
            raise ValueError("Saved timestamps must be timezone-naive and nonmissing.")
    numeric = ["Horizon", "Actual_Count", "Predicted_Count", "Error", "Absolute_Error"]
    forecasts[numeric] = forecasts[numeric].apply(pd.to_numeric, errors="raise")
    if (not np.isfinite(forecasts[numeric].to_numpy()).all()
            or forecasts[["Actual_Count", "Predicted_Count"]].lt(0).any().any()
            or set(forecasts.Strategy) != set(STRATEGIES)
            or forecasts.duplicated(["Origin", "Strategy", "Horizon"]).any()):
        raise ValueError("Saved predictions contain invalid counts or duplicate forecast cases.")
    n_origins = settings["origins_evaluated"]
    origins = pd.DatetimeIndex(sorted(forecasts.Origin.unique()))
    if (len(origins) != n_origins or not origins.equals(pd.DatetimeIndex(pd.to_datetime(settings["origins"])))
            or len(forecasts) != n_origins * HOURS * len(STRATEGIES)
            or settings.get("forecasts_per_strategy") != n_origins * HOURS
            or not forecasts.groupby(["Origin", "Strategy"]).Horizon.apply(lambda x: set(x) == set(range(1, 25))).all()
            or not forecasts.Datetime.eq(forecasts.Origin + pd.to_timedelta(forecasts.Horizon, unit="h")).all()
            or not forecasts.Datetime.lt(CUTOFF).all()
            or not forecasts.Origin.eq(forecasts.Origin.dt.floor("h")).all()):
        raise ValueError("Saved forecast counts, timestamps or cutoff disagree with the metadata.")
    outcomes = forecasts.pivot(index=["Origin", "Horizon", "Datetime"], columns="Strategy", values="Actual_Count")
    if outcomes.isna().any().any() or outcomes.nunique(axis=1).ne(1).any():
        raise ValueError("Strategies were not evaluated on identical outcomes.")
    errors = forecasts.Predicted_Count - forecasts.Actual_Count
    if not np.allclose(errors, forecasts.Error) or not np.allclose(errors.abs(), forecasts.Absolute_Error):
        raise ValueError("Saved errors do not match the saved predictions.")
    windows = pd.read_csv(directory / "validation_training_windows.csv", parse_dates=["Origin", "Training_Start", "Training_End"])
    if (len(windows) != n_origins or windows.Origin.duplicated().any()
            or not pd.DatetimeIndex(sorted(windows.Origin)).equals(origins)
            or not windows.Training_End.lt(windows.Origin).all()
            or not windows.Training_Start.le(windows.Training_End).all()):
        raise ValueError("Training-window chronology is incompatible with this experiment.")
    tables, thresholds = diagnostic_tables(forecasts)
    if any(not np.isclose(settings["demand_thresholds"][key], value) for key, value in thresholds.items()):
        raise ValueError("Saved retrospective demand thresholds disagree with the outcomes.")
    # Check saved metric tables against recomputation; display the actual saved tables.
    for name, calculated in tables.items():
        saved = pd.read_csv(directory / f"validation_{name}.csv")
        if "Origin" in saved:
            saved["Origin"] = pd.to_datetime(saved.Origin, errors="raise")
        if len(saved) != len(calculated) or set(calculated.columns).difference(saved.columns):
            raise ValueError(f"Incomplete saved metric table: {name}.")
        for column in calculated:
            if pd.api.types.is_numeric_dtype(calculated[column]):
                if not np.allclose(saved[column].to_numpy(dtype=float), calculated[column].to_numpy(dtype=float),
                                rtol=1e-7, atol=1e-7, equal_nan=True):
                    raise ValueError(f"Saved metrics disagree with predictions: {name}/{column}.")
            elif not np.array_equal(saved[column].astype(str).to_numpy(), calculated[column].astype(str).to_numpy()):
                raise ValueError(f"Saved metric grouping differs from predictions: {name}/{column}.")
        tables[name] = saved
    return settings, tables, forecasts


def discover_experiments() -> tuple[list[dict], list[str]]:
    """Discover compatible complete runs; sort by saved metadata modification time."""
    runs, skipped = [], []
    compatibility = method_version()
    for directory in OUTPUT_PATH.glob("forecast_validation_*"):
        if not directory.is_dir():
            continue
        try:
            settings = json.loads((directory / "validation_settings.json").read_text(encoding="utf-8"))
            check_metadata(settings)
            signature = experiment_signature(directory)
            load_experiment(str(directory), signature, compatibility)
            runs.append(dict(folder=str(directory), name=directory.name, signature=signature,
                            modified=(directory / "validation_settings.json").stat().st_mtime_ns,
                            origins=settings["origins_evaluated"], stride=settings["arguments"]["stride_days"]))
        except ERRORS as error:
            skipped.append(f"{directory.name}: {error}")
    return sorted(runs, key=lambda run: (run["modified"], run["name"]), reverse=True), skipped


def apply_style() -> None:
    """Apply restrained local styling without external fonts or assets."""
    st.markdown("""<style>
    .stApp {background:#f6f8fa; color:#172c3a;}
    .block-container {padding-top:2.2rem; padding-bottom:3rem; max-width:1400px;}
    h1,h2,h3 {font-family:Inter,Segoe UI,sans-serif; letter-spacing:-.035em; color:#172c3a;}
    [data-testid="stSidebar"] {background:#eaf0f3; border-right:1px solid #dbe4e8;}
    [data-testid="stMetric"] {background:white; border:1px solid #dce5e9; border-radius:12px; padding:18px 20px;}
    [data-testid="stMetricLabel"] {color:#536977;}
    [data-testid="stMetricValue"] {font-size:1.8rem; font-weight:650;}
    .eyebrow {font-size:.72rem; font-weight:750; letter-spacing:.18em; color:#087f8c; margin-bottom:12px;}
    .brand {font-size:1.65rem; font-weight:800; color:#172c3a; letter-spacing:-.06em; margin-bottom:0;}
    .subtle {color:#617787; font-size:.9rem; line-height:1.6;}
    .stButton button[kind="primary"] {background:#087f8c; border-color:#087f8c; border-radius:9px;}
    </style>""", unsafe_allow_html=True)


def chart(figure: go.Figure, key: str, height: int = 370) -> None:
    """Render a responsive, accessible Plotly chart with consistent labels."""
    figure.update_layout(template="plotly_white", height=height,
                        font=dict(family="Segoe UI, Arial, sans-serif", color="#304957", size=13),
                        margin=dict(l=15, r=15, t=35, b=20), paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="white", legend=dict(orientation="h", y=1.16, x=0),
                        hovermode="x unified")
    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(gridcolor="#e8eef2", zerolinecolor="#cbd7df")
    st.plotly_chart(figure, width="stretch", key=key, config={"displaylogo": False, "scrollZoom": False})


def heading(title: str, description: str) -> None:
    """Render a consistent page introduction."""
    st.markdown('<div class="eyebrow">JETRAIL / DEMAND LAB</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(description)


def archive() -> pd.DataFrame:
    """Load historical observations for descriptive pages, separately from forecasts."""
    return historical_data(fingerprint(DATA_PATH))


def overview_page() -> None:
    """Describe the full historical archive using explicitly labeled hourly averages."""
    heading("Historical demand, in perspective.",
            "Explore passenger patterns, reconstruct a 24-hour forecast, and examine how it behaves across time.")
    data = archive()
    columns = st.columns(3)
    columns[0].metric("Observed hourly counts", f"{data.Count.notna().sum():,}")
    columns[1].metric("Average hourly demand", f"{data.Count.mean():,.1f}", help="Mean of nonmissing hourly passenger counts.")
    columns[2].metric("Maximum hourly demand", f"{data.Count.max():,.0f}")
    st.caption(f"First recorded hour: {data.index.min():%d %b %Y, %H:%M} · "
            f"Last recorded hour: {data.index.max():%d %b %Y, %H:%M} · "
            f"Missing hours/counts on the hourly grid: {data.Count.isna().sum():,}")
    st.subheader("A growing demand profile")
    daily = data.Count.resample("D").mean()
    fig = go.Figure()
    fig.add_scatter(x=daily.index, y=daily, name="Daily mean of hourly counts", line=dict(color="#a5cbd0", width=1))
    fig.add_scatter(x=daily.index, y=daily.rolling(7, min_periods=7).mean(), name="Trailing 7-day mean", line=dict(color="#087f8c", width=2.5))
    fig.update_layout(yaxis_title="Passengers per hour (mean)", xaxis_title="Historical date")
    chart(fig, "overview_trend", 400)
    left, right = st.columns([1.3, 1])
    with left:
        st.subheader("About this project")
        st.write("The historical JetRail forecasting challenge provides hourly passenger counts. "
                "This portfolio follows the full workflow: exploration, causal feature engineering, "
                "baseline comparisons, demand-shift diagnostics, and rolling-origin validation.")
        st.write("The forecasting strategy predicts an hourly change with an expanding Random Forest, "
                "then feeds each reconstructed count into the next forecast hour.")
    with right:
        st.info("Historical demonstration\n\nThe archive ends in September 2014. These are historical passenger counts, "
                "not a live railway feed. Forecast demonstrations and model comparisons use pre-test development data.")
    st.caption("This descriptive overview includes the full archive, including the previously inspected test period. "
            "Browsing those observations does not make them independent evaluation data.")


def aggregate_demand(data: pd.DataFrame, resolution: str, statistic: str) -> pd.DataFrame:
    """Keep empty bins missing and distinguish sums from means in the explorer."""
    if resolution == "Hourly":
        return data.rename(columns={"Count": "Demand"}).reset_index()
    frequency = {"Daily": "D", "Monthly": "MS"}[resolution]
    grouped = data.Count.resample(frequency)
    values = grouped.sum(min_count=1) if statistic == "Total passengers" else grouped.mean()
    return values.rename("Demand").to_frame().join(grouped.count().rename("Observed_Hours")).reset_index()


def explorer_page() -> None:
    """Explore selected dates, aggregation levels, calendar averages and distributions."""
    heading("Explore the historical archive.", "Inspect demand at different time scales. All charts describe observed history.")
    data = archive()
    selected = st.date_input("Historical date range", value=(data.index.min().date(), data.index.max().date()),
                            min_value=data.index.min().date(), max_value=data.index.max().date())
    if not isinstance(selected, (tuple, list)) or len(selected) != 2:
        st.info("Choose both a start date and an end date.")
        return
    start, end = selected
    part = data.loc[(data.index >= pd.Timestamp(start)) & (data.index < pd.Timestamp(end) + pd.Timedelta(days=1))]
    if part.empty or not part.Count.notna().any():
        st.warning("No observed counts are available in this date range.")
        return
    a, b = st.columns(2)
    resolution = a.selectbox("Time resolution", ["Hourly", "Daily", "Monthly"])
    statistic = b.selectbox("Aggregation", ["Total passengers", "Average hourly demand"], disabled=resolution == "Hourly")
    aggregated = aggregate_demand(part, resolution, statistic)
    label = "Passengers per hour"
    if resolution != "Hourly":
        period = "day" if resolution == "Daily" else "month"
        label = f"Passengers per {period} (sum)" if statistic == "Total passengers" else "Passengers per hour (mean)"
    st.subheader(f"{resolution} demand")
    fig = px.line(aggregated, x="Datetime", y="Demand", labels={"Demand": label}, color_discrete_sequence=["#087f8c"])
    chart(fig, "explorer_series")
    st.caption("Hourly shows individual counts. Daily/monthly totals sum available hours; averages remain passengers per hour. "
            "Missing hours are not imputed, and boundary periods or periods containing gaps can have incomplete totals.")
    left, right = st.columns(2)
    with left:
        st.subheader("Average by hour")
        hourly = part.groupby(part.index.hour).Count.mean().reindex(range(24)).rename("Mean_Count").reset_index()
        hourly.columns = ["Hour", "Mean_Count"]
        chart(px.bar(hourly, x="Hour", y="Mean_Count", labels={"Mean_Count": "Mean passengers per hour"},
                    color_discrete_sequence=["#087f8c"]), "explorer_hour", 320)
    with right:
        st.subheader("Average by weekday")
        weekday = part.groupby(part.index.day_name()).Count.mean().reindex(WEEKDAYS).rename("Mean_Count").reset_index()
        weekday.columns = ["Weekday", "Mean_Count"]
        chart(px.bar(weekday, x="Weekday", y="Mean_Count", labels={"Mean_Count": "Mean passengers per hour"},
                    color_discrete_sequence=["#7586b5"]), "explorer_weekday", 320)
    st.subheader("Distribution of hourly counts")
    chart(px.histogram(part.reset_index(), x="Count", nbins=45, labels={"Count": "Passengers per hour"},
                    color_discrete_sequence=["#087f8c"]), "explorer_distribution", 300)
    st.caption("Calendar averages pool the selected hours. Growth, different coverage, and changing calendar mix can affect them.")
    with st.expander("Inspect aggregated data"):
        st.dataframe(aggregated, hide_index=True, width="stretch")


def forecast_page() -> None:
    """Generate forecasts on explicit request; revealing outcomes never triggers a refit."""
    heading("Forecast the next 24 historical hours.",
            "Choose a historical origin and generate a fixed-origin recursive forecast with RF Delta Expanding.")
    version = fingerprint(DEVELOPMENT_PATH)
    algorithm = method_version()
    history = development_data(version)
    origins = eligible_origins(history)
    if origins.empty:
        st.warning("No eligible forecast origins. At least 180 days of history, 168 complete context hours, "
                "and 24 observed pre-test outcome hours are required.")
        return
    st.info(f"Development-only demonstration · all forecast targets precede {CUTOFF:%d %b %Y, %H:%M}. "
            "Training labels strictly precede the origin. The observed origin count supplies context.")
    col1, col2, col3 = st.columns([2, 1, 2])
    day = col1.date_input("Forecast origin date", value=origins[-1].date(),
                        min_value=origins[0].date(), max_value=origins[-1].date())
    available = origins[origins.date == day]
    if available.empty:
        st.warning("This date has incomplete context or outcomes. Choose another historical date.")
        return
    origin = col2.selectbox("Origin hour", list(available), format_func=lambda stamp: stamp.strftime("%H:%M"))
    col3.caption("One fit per origin. Predictions update the working history for all 24 hours; the forest is not refitted inside the horizon.")
    token = (str(origin), version, algorithm, json.dumps(RF_PARAMS, sort_keys=True))
    if st.button("Generate 24-hour forecast", type="primary", key="generate_forecast"):
        with st.spinner("Validating features, fitting the historical delta model, and forecasting 24 hours…"):
            forecast, training = generate_forecast(*token)
        st.session_state["forecast_result"] = dict(token=token, forecast=forecast, training=training)
    result = st.session_state.get("forecast_result")
    if result is None or result["token"] != token:
        st.caption("Generate a forecast for this origin. Changing controls does not train a model automatically.")
        return
    forecast, training = result["forecast"], result["training"]
    st.success(f"24 hourly predictions ready · {training['Training_Observations']:,} training rows · "
            f"training ends {pd.Timestamp(training['Training_End']):%d %b %Y, %H:%M}.")
    reveal = st.checkbox("Reveal actual historical outcomes", value=False, key="reveal_actuals",
                        help="Outcomes are joined after prediction, solely for retrospective comparison.")
    context = history.loc[origin - pd.Timedelta(hours=167):origin]
    fig = go.Figure()
    fig.add_scatter(x=context.index, y=context.Count, name="Observed context · last 7 days", line=dict(color="#8fa3b0", width=1.3))
    fig.add_scatter(x=[origin, *forecast.Datetime], y=[context.Count.iloc[-1], *forecast.Predicted_Count],
                name="RF Delta · recursive forecast", line=dict(color="#087f8c", width=2.7))
    fig.add_vline(x=origin.to_pydatetime(), line_dash="dot", line_color="#9a6b32")
    display = forecast.copy()
    if reveal:
        display["Actual_Count"] = history.Count.reindex(pd.DatetimeIndex(display.Datetime)).to_numpy()
        fig.add_scatter(x=display.Datetime, y=display.Actual_Count, name="Actual outcomes · revealed after forecasting",
                    line=dict(color="#a66d2f", width=2, dash="dash"))
    fig.update_layout(yaxis_title="Passengers per hour", xaxis_title="Historical timestamp")
    chart(fig, "forecast_plot", 450)
    st.caption(f"Dotted line: forecast origin {origin}. Everything to its right is generated without future actuals. "
            "Fractional predictions are retained; negative reconstructed counts are clipped to zero.")
    if reveal:
        scores = metrics(display.Actual_Count.to_numpy(), display.Predicted_Count.to_numpy())
        columns = st.columns(3)
        for col, key in zip(columns, ("MAE", "RMSE", "MBE")):
            col.metric(f"This origin · {key}", f"{scores[key]:.2f}")
        st.caption("These scores describe this selected historical origin only. MBE = predicted minus actual.")
    st.subheader("Hourly forecast")
    st.dataframe(display[["Datetime", "Horizon", "Predicted_Count", *(["Actual_Count"] if reveal else [])]],
                hide_index=True, width="stretch", column_config={"Predicted_Count": st.column_config.NumberColumn("Predicted passengers", format="%.2f")})
    st.download_button("Download forecast CSV", forecast.to_csv(index=False).encode("utf-8"),
                    file_name=f"jetrail_forecast_{origin:%Y%m%d_%H%M}.csv", mime="text/csv")
    with st.expander("Forecast method and training audit"):
        st.write("Calls fit_delta_model() and recursive_forecast() from forecast_24h.py. "
                "No saved absolute-demand model is loaded. Cached predictions are keyed by origin, "
                "dataset contents, forecasting source, feature order, library versions and RF settings.")
        st.json({**training, "Training_Start": str(training["Training_Start"]),
                "Training_End": str(training["Training_End"]), "RF_parameters": RF_PARAMS, "Feature_order": FEATURE_COLUMNS})


def performance_page() -> None:
    """Present only validated, compatible, completed retrospective experiments."""
    heading("Performance across historical origins.",
            "Compare the established strategy against seasonal naive and recursive persistence on identical outcomes.")
    with st.spinner("Discovering completed experiments and checking saved metrics…"):
        runs, skipped = discover_experiments()
    if not runs:
        st.warning("No compatible completed forecast_validation experiment was found. "
                "Run python forecast_validation.py from the project directory.")
        if skipped:
            with st.expander("Unavailable experiment details"):
                st.write("\n\n".join(skipped))
        return
    folders = [run["folder"] for run in runs]
    lookup = {run["folder"]: run for run in runs}
    selected = st.selectbox("Completed experiment", folders,
                            format_func=lambda folder: f"{lookup[folder]['name']} · {lookup[folder]['origins']} origins · {lookup[folder]['stride']}-day stride")
    run = lookup[selected]
    settings, tables, predictions = load_experiment(selected, run["signature"], method_version())
    st.caption("Default: latest compatible run by saved metadata modification time. Only experiments marked complete "
            "and consistent with their saved predictions are listed.")
    rf = tables["overall_summary"].set_index("Strategy").loc[STRATEGIES[0]]
    columns = st.columns(4)
    for col, field, label in zip(columns, ("MAE", "RMSE", "WAPE_pct", "MBE"),
                                ("RF · pooled MAE", "RF · pooled RMSE", "RF · WAPE", "RF · mean bias")):
        value = f"{rf[field]:.2f}{'%' if field == 'WAPE_pct' else ''}" if np.isfinite(rf[field]) else "Undefined"
        col.metric(label, value,
                help="Calculated from individual forecast errors. Bias = predicted minus actual; negative means underprediction.")
    comparisons = tables["baseline_comparisons"]
    wins = int(comparisons.RF_Beats_Seasonal.sum())
    st.write(f"**{settings['origins_evaluated']} origins · {int(rf.Forecasts):,} forecasts per strategy · "
            f"RF beats seasonal naive at {wins}/{len(comparisons)} origins ({100*wins/len(comparisons):.2f}%).**")
    weekdays = [day for day in WEEKDAYS if day in predictions.Datetime.dt.day_name().unique()]
    st.warning(f"Retrospective development evaluation. Target weekdays sampled: {', '.join(weekdays)}. "
            f"Origin stride: {settings['arguments']['stride_days']} day(s). "
            "A weekly stride fixes weekday composition; missing weekdays or weekends cannot be assessed. "
            "Adjacent windows and shared expanding training history create dependence.")
    summary = tables["overall_summary"].copy()
    chart(px.bar(summary, x="Strategy", y="MAE", color="Strategy", color_discrete_map=COLORS,
                labels={"MAE": "Pooled MAE (passengers)"}), "performance_overall", 320)
    st.dataframe(summary.rename(columns={"MBE": "Mean bias", "WAPE_pct": "WAPE (%)"}), hide_index=True, width="stretch")
    left, right = st.columns(2)
    with left:
        st.subheader("Error by forecast horizon")
        metric = st.radio("Horizon metric", ["MAE", "RMSE", "MBE"], horizontal=True)
        fig = px.line(tables["metrics_by_horizon"], x="Horizon", y=metric, color="Strategy", color_discrete_map=COLORS,
                    labels={"Horizon": "Hours after origin", metric: "Bias (predicted − actual)" if metric == "MBE" else f"{metric} (passengers)"})
        if metric == "MBE":
            fig.add_hline(y=0, line_dash="dot")
        chart(fig, "performance_horizon", 360)
    with right:
        st.subheader("Error across origins")
        chart(px.line(tables["metrics_by_origin"], x="Origin", y="MAE", color="Strategy", color_discrete_map=COLORS,
                    labels={"MAE": "MAE (passengers)"}), "performance_origins", 360)
    st.subheader("Where demand is highest")
    demand = tables["metrics_by_demand"]
    chart(px.bar(demand, x="Demand_Level", y="MAE", color="Strategy", barmode="group", color_discrete_map=COLORS,
                category_orders={"Demand_Level": ["Low", "Middle", "High"]},
                labels={"Demand_Level": "Retrospective actual-demand group", "MAE": "MAE (passengers)"}), "performance_demand", 330)
    st.dataframe(demand.rename(columns={"MBE": "Mean bias", "WAPE_pct": "WAPE (%)"}), hide_index=True, width="stretch")
    thresholds = settings["demand_thresholds"]
    st.caption(f"Low demand ≤ {thresholds['low_max_inclusive']:g}; high demand > {thresholds['high_min_exclusive']:g}. "
            "Quartile thresholds use evaluated actual outcomes after forecasting, only for diagnostics. "
            "Demand changes over time; compare WAPE and bias alongside absolute errors.")
    with st.expander("Sampling, reproducibility and evaluation limits"):
        st.write(DEPENDENCE)
        st.write(settings["disclaimer"])
        st.write(f"Origin range: {predictions.Origin.min()} → {predictions.Origin.max()}.")
        st.json(settings)
    if skipped:
        with st.expander(f"Skipped {len(skipped)} incomplete or incompatible experiment(s)"):
            st.write("\n\n".join(skipped))
    st.download_button("Download experiment summary", summary.to_csv(index=False).encode("utf-8"),
                    file_name=f"{Path(selected).name}_summary.csv", mime="text/csv")


def insights_page() -> None:
    """Explain the tested method and read prior drift diagnostics without loading models."""
    heading("Understand the forecasting method.", "A practical guide to the decisions behind this historical demand experiment.")
    concepts = [
        ("Time series forecasting", "Predict future values from a time-ordered history. Chronology matters: a model must never see observations that were unavailable when its forecast was issued."),
        ("Lag features", "Lag 1 is the previous hour; lags 24 and 168 describe the same hour yesterday and last week. The pipeline also uses lags 2, 3 and 48, plus calendar descriptors."),
        ("Rolling statistics", "Trailing means and sample standard deviations summarize the previous 24 and 168 hours. The 24-hour minimum and maximum describe the recent range. Counts are shifted before rolling, excluding the target hour."),
        ("Random Forest regression", "An ensemble of decision trees averages predictions learned from historical feature patterns. The fixed configuration uses 200 trees, depth 15, and at least two observations per leaf. Trees do not automatically extrapolate a rising trend."),
        ("Delta-target forecasting", "The target is Count − lag_1: an hourly change. Add the predicted change to the previous working count and clip the result at zero. This differs from the earlier absolute-demand Random Forest saved by train.py."),
        ("Recursive multi-step prediction", "Fit once at the origin and repeat 24 times: calculate features, predict a change, reconstruct a count, and append it. Later lags and rolling windows use predictions. Errors can propagate through the working history."),
        ("Data drift", "The distribution of demand or model inputs may change over time. Rising demand can move later observations outside earlier ranges. Distribution shift alone does not establish a change in the conditional relationship, or concept drift."),
        ("Walk-forward validation", "At each chronological origin, refit with eligible earlier labels and forecast the next 24 hours. Compare strategies on matching outcomes. Overlapping training sets and serially correlated demand mean origin scores are dependent."),
    ]
    for start in range(0, len(concepts), 2):
        columns = st.columns(2)
        for col, (title, explanation) in zip(columns, concepts[start:start + 2]):
            with col.container(border=True):
                st.subheader(title)
                st.write(explanation)
    st.subheader("Previously observed upward demand shift")
    path = OUTPUT_PATH / "drift_target_statistics.csv"
    if path.is_file():
        try:
            stats = pd.read_csv(path)
            if {"Split", "Mean", "Maximum", "Observations"}.difference(stats.columns):
                raise ValueError("Prior drift summary has an incomplete schema.")
            values = stats.set_index("Split")
            if not {"Training", "Validation", "Test"}.issubset(values.index):
                raise ValueError("Prior drift summary does not contain all three splits.")
            means = pd.to_numeric(values.loc[["Training", "Validation", "Test"], "Mean"], errors="raise")
            if not np.isfinite(means).all():
                raise ValueError("Prior drift summary has invalid means.")
            chart(px.bar(stats, x="Split", y="Mean", color="Split",
                        color_discrete_sequence=["#087f8c", "#7586b5", "#ad7b3e"],
                        labels={"Mean": "Mean hourly passenger count"}), "insights_drift", 300)
            st.write(f"The existing diagnostic reports mean hourly counts of **{means.iloc[0]:.1f}** in training, "
                    f"**{means.iloc[1]:.1f}** in validation, and **{means.iloc[2]:.1f}** in the previously inspected test split. "
                    "Later demand was substantially higher, making the early training distribution less representative.")
            st.caption("Source: saved drift_target_statistics.csv. This is an existing retrospective diagnostic, "
                    "not a new test evaluation or a basis for tuning this dashboard's model.")
        except ERRORS as error:
            st.info(f"Prior drift summary is unavailable: {error}")
    else:
        st.info("The project previously observed an upward target-distribution shift. "
                "The saved drift_target_statistics.csv is unavailable, so numerical comparisons are omitted.")
    st.subheader("Four evaluation concepts to keep separate")
    st.dataframe(pd.DataFrame({
        "Concept": ["Rolling one-step prediction", "Fixed-origin recursive 24-hour forecast", "Historical development evaluation", "Independent future performance"],
        "Meaning": ["Actual counts through t−1 become available before every prediction.",
                    "One origin; no later actuals are used as inputs for any of the 24 hours.",
                    "Retrospective results used in a project with prior model selection and diagnostics.",
                    "Requires genuinely new unseen chronological data; has not been established here."]}),
        hide_index=True, width="stretch")


def main() -> None:
    """Render a read-only historical dashboard; fitting occurs only on explicit request."""
    st.set_page_config(page_title="JetRail | Demand Lab", page_icon="🚆", layout="wide")
    apply_style()
    with st.sidebar:
        st.markdown('<p class="brand">JetRail<span style="color:#087f8c">.</span></p>', unsafe_allow_html=True)
        st.markdown('<p class="subtle">PASSENGER DEMAND LAB<br>Historical forecasting portfolio</p>', unsafe_allow_html=True)
        st.divider()
        page = st.radio("Workspace", PAGES, label_visibility="collapsed", key="navigation")
        st.divider()
        st.caption("ARCHIVE · AUG 2012—SEP 2014")
        st.caption("24-hour recursive forecasts\n\nExpanding Random Forest · delta target")
        st.info("Historical demonstration. No live data or independent future accuracy guarantee.")
    st.caption("HISTORICAL JETRAIL CHALLENGE · RETROSPECTIVE DEVELOPMENT · NOT A LIVE RAILWAY SYSTEM")
    renderers = dict(zip(PAGES, [overview_page, explorer_page, forecast_page, performance_page, insights_page]))
    try:
        renderers[page]()
    except ERRORS as error:
        st.error(f"This page could not load: {error}")
        st.info("Check the project's data and output files. Run features.py if the saved feature table is missing; "
                "run forecast_validation.py if completed validation artifacts are missing. Existing data and models are not overwritten by this app.")
    st.divider()
    st.caption("JetRail Demand Lab · Historical analysis, explicit forecast assumptions, and reproducible evaluation. "
            "Fully independent validation requires new unseen chronological data.")


if __name__ == "__main__":
    main()
