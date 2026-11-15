"""Recursive fixed-origin 24-hour forecasts on historical development data.

Run python forecast_24h.py --help. These retrospective development evaluations
are not independent final test performance. No original test counts are used.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from uuid import uuid4

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor

from backtesting import CUTOFF, RF_PARAMS
from features import FEATURE_COLUMNS, LAGS, add_features, validate_time_index

BASE = Path(__file__).resolve().parent
HOURS = 24
CONTEXT = max(168, *LAGS)
STRATEGIES = ("RF delta expanding", "Recursive persistence", "Seasonal naive 24h")
DISCLAIMER = (
    "Retrospective pre-test development evaluation, informed by earlier model "
    "selection and diagnostics; not independent final test performance. "
    "No tuning or model selection is performed here. Errors can accumulate "
    "recursively; one-step backtest scores are not directly comparable."
)


def load_development_history() -> pd.DataFrame:
    """Filter before inspecting counts/features, preserve gaps, audit saved features."""
    source = pd.read_csv(BASE / "data" / "passenger_features.csv")
    required = ["Datetime", "Count", *FEATURE_COLUMNS]
    if set(required).difference(source.columns):
        raise ValueError("Saved feature table has an incomplete schema.")
    source["Datetime"] = pd.to_datetime(source["Datetime"], errors="raise")
    source = source.loc[source.Datetime < CUTOFF, required].set_index("Datetime")
    validate_time_index(source, require_hourly=False)
    source = source.apply(pd.to_numeric, errors="raise")
    if np.isinf(source.to_numpy(dtype=float)).any() or source.Count.lt(0).any():
        raise ValueError("Development data contain infinity or negative counts.")
    grid = pd.date_range(source.index.min(), source.index.max(), freq="h", name="Datetime")
    history = source[["Count"]].reindex(grid)
    expected = add_features(history).reindex(source.index)[FEATURE_COLUMNS]
    np.testing.assert_allclose(source[FEATURE_COLUMNS].to_numpy(dtype=float),
                               expected.to_numpy(dtype=float), rtol=1e-7,
                               atol=1e-7, equal_nan=True)
    print("Saved features verified against the existing feature pipeline.")
    return history


def next_features(working: pd.DataFrame) -> pd.DataFrame:
    """Use the exact pipeline on the last 168 hours plus an unknown next target."""
    validate_time_index(working)
    if len(working) < CONTEXT:
        raise ValueError(f"Forecast requires at least {CONTEXT} hours of context.")
    next_time = working.index[-1] + pd.Timedelta(hours=1)
    extended = working.iloc[-CONTEXT:][["Count"]].reindex(
        pd.date_range(working.index[-CONTEXT], next_time, freq="h", name="Datetime"))
    row = add_features(extended).iloc[[-1]][FEATURE_COLUMNS]
    if not np.isfinite(row.to_numpy(dtype=float)).all():
        raise ValueError("Forecast requires 168 consecutive observed hours at the origin.")
    return row


def fit_delta_model(history: pd.DataFrame, origin: pd.Timestamp) -> tuple[RandomForestRegressor, dict]:
    """Fit expanding history strictly before origin using Count minus lag_1."""
    train = add_features(history.loc[history.index < origin]).dropna(subset=["Count", *FEATURE_COLUMNS])
    if len(train) < 2:
        raise ValueError("Too few complete training observations.")
    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(train[FEATURE_COLUMNS], train.Count - train.lag_1)
    return model, dict(Training_Start=train.index.min(), Training_End=train.index.max(),
                      Training_Observations=len(train))


def recursive_forecast(model: RandomForestRegressor, observed: pd.DataFrame) -> pd.DataFrame:
    """Forecast 24 changes, append clipped counts, and recompute historical features."""
    working = observed[["Count"]].copy()
    rows = []
    for horizon in range(1, HOURS + 1):
        features = next_features(working)
        delta = float(model.predict(features)[0])
        previous = float(working.Count.iloc[-1])
        count = float(np.clip(previous + delta, 0, None))
        if not np.isfinite(count):
            raise ValueError("Nonfinite recursive prediction.")
        timestamp = features.index[0]
        rows.append(dict(Datetime=timestamp, Horizon=horizon, Predicted_Count=count,
                         Predicted_Delta=delta, Previous_Working_Count=previous))
        working.loc[timestamp, "Count"] = count
    return pd.DataFrame(rows)


def validate_recursive_features() -> None:
    """Check bounded context against full pipeline, predicted feedback and isolation."""
    index = pd.date_range("2020-01-01", periods=430, freq="h", name="Datetime")
    source = pd.DataFrame({"Count": 100 + np.arange(len(index)) % 57}, index=index, dtype=float)
    working = source.iloc[:400].copy()
    for step in range(HOURS):
        row = next_features(working)
        timestamp = row.index[0]
        extended = working.reindex(working.index.append(pd.DatetimeIndex([timestamp])))
        expected = add_features(extended).iloc[[-1]][FEATURE_COLUMNS]
        np.testing.assert_allclose(row.to_numpy(dtype=float), expected.to_numpy(dtype=float),
                                   rtol=1e-10, atol=1e-10)
        # Changing current/future actuals cannot change forecast-time features.
        changed = source.copy()
        changed.loc[changed.index >= timestamp, "Count"] = 999999
        changed.loc[working.index, "Count"] = working.Count
        np.testing.assert_allclose(row.to_numpy(dtype=float),
            add_features(changed).loc[[timestamp], FEATURE_COLUMNS].to_numpy(dtype=float))
        if float(row.lag_1.iloc[0]) != float(working.Count.iloc[-1]):
            raise AssertionError("Recursive lag did not use the working sequence.")
        working.loc[timestamp, "Count"] = float(17 + step * 3.5)
    print("24-step feature parity, recursive feedback and future-target isolation checks passed.")


def choose_origins(history: pd.DataFrame, count: int, spacing_days: int,
                   min_history_days: int, explicit: list[str] | None) -> list[pd.Timestamp]:
    """Choose chronological development origins with complete 24-hour outcomes."""
    if min(count, spacing_days, min_history_days) <= 0:
        raise ValueError("Origin count, spacing and minimum history must be positive.")
    last = min(CUTOFF - pd.Timedelta(hours=HOURS + 1), history.index.max() - pd.Timedelta(hours=HOURS))
    origins = (sorted(pd.Timestamp(value) for value in explicit) if explicit else
               [last - pd.Timedelta(days=spacing_days * i) for i in reversed(range(count))])
    if len(set(origins)) != len(origins):
        raise ValueError("Origins must be unique.")
    for origin in origins:
        if pd.isna(origin) or origin.tzinfo is not None or origin != origin.floor("h"):
            raise ValueError("Origins must be timezone-naive exact clock hours.")
        if origin >= CUTOFF or origin + pd.Timedelta(hours=HOURS) >= CUTOFF:
            raise ValueError("Origin and all forecast targets must precede the original test cutoff.")
        if origin - history.index.min() < pd.Timedelta(days=min_history_days):
            raise ValueError("Origin has insufficient chronological training history.")
        required = pd.date_range(origin - pd.Timedelta(hours=CONTEXT - 1),
                                 origin + pd.Timedelta(hours=HOURS), freq="h")
        if history.reindex(required).Count.isna().any():
            raise ValueError(f"Missing origin context or evaluation actuals at {origin}.")
    return origins


def evaluate(history: pd.DataFrame, origins: list[pd.Timestamp]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forecast from sliced histories; attach future actuals only after prediction."""
    forecasts, training = [], []
    for origin in origins:
        observed = history.loc[:origin].copy()
        model, metadata = fit_delta_model(observed, origin)
        rf = recursive_forecast(model, observed)
        training.append(dict(Origin=origin, **metadata))
        for strategy in STRATEGIES:
            frame = rf.copy()
            if strategy != STRATEGIES[0]:
                frame[["Predicted_Delta", "Previous_Working_Count"]] = np.nan
                frame["Predicted_Count"] = (float(observed.Count.iloc[-1])
                    if strategy == STRATEGIES[1] else observed.Count.iloc[-HOURS:].to_numpy())
            frame["Origin"] = origin
            frame["Strategy"] = strategy
            frame["Actual_Count"] = history.Count.reindex(pd.DatetimeIndex(frame.Datetime)).to_numpy()
            frame["Residual"] = frame.Actual_Count - frame.Predicted_Count
            forecasts.append(frame)
        print(f"Origin {origin}: fitted {metadata['Training_Observations']:,} rows strictly before origin; "
              "completed 24 fixed-origin hours.", flush=True)
    return pd.concat(forecasts, ignore_index=True), pd.DataFrame(training)


def metric_table(forecasts: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    """Compute MAE and pooled RMSE from individual errors in each group."""
    rows = []
    for keys, part in forecasts.groupby(groups, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        error = part.Predicted_Count.to_numpy() - part.Actual_Count.to_numpy()
        rows.append(dict(zip(groups, keys), Observations=len(part),
                         MAE=float(np.abs(error).mean()), RMSE=float(np.sqrt(np.mean(error ** 2)))))
    return pd.DataFrame(rows)


def save_figures(forecasts: pd.DataFrame, horizon_metrics: pd.DataFrame,
                 overall: pd.DataFrame, directory: Path) -> None:
    """Save horizon errors, overall errors and the chronological middle origin."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for strategy, part in horizon_metrics.groupby("Strategy", sort=False):
        for ax, metric in zip(axes, ("MAE", "RMSE")):
            ax.plot(part.Horizon, part[metric], label=strategy)
            ax.set(xlabel="Forecast horizon (hours)", ylabel=f"{metric} (passengers)", xticks=[1, 6, 12, 18, 24])
            ax.grid(alpha=.25)
    axes[0].legend()
    fig.suptitle("Historical development: fixed-origin 24-hour errors")
    fig.tight_layout()
    fig.savefig(directory / "horizon_errors.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    overall.set_index("Strategy")[["MAE", "RMSE"]].plot.bar(ax=ax, rot=0)
    ax.set(title="Historical development: pooled 24-hour errors", ylabel="Passengers", xlabel="")
    fig.tight_layout()
    fig.savefig(directory / "overall_errors.png", dpi=160)
    plt.close(fig)
    origins = sorted(forecasts.Origin.unique())
    origin = origins[len(origins) // 2]
    part = forecasts.loc[forecasts.Origin.eq(origin)]
    fig, ax = plt.subplots(figsize=(12, 5))
    actual = part.loc[part.Strategy.eq(STRATEGIES[0])]
    ax.plot(actual.Horizon, actual.Actual_Count, color="black", label="Actual", linewidth=2)
    for strategy, frame in part.groupby("Strategy", sort=False):
        ax.plot(frame.Horizon, frame.Predicted_Count, label=strategy)
    ax.set(title=f"Historical development forecast from {pd.Timestamp(origin)}",
           xlabel="Hours after origin", ylabel="Passengers")
    ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "example_forecast.png", dpi=160)
    plt.close(fig)


def main() -> int:
    """Validate, evaluate and save a unique run without changing earlier artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origins", type=int, default=6, help="Number of default historical origins")
    parser.add_argument("--spacing-days", type=int, default=30)
    parser.add_argument("--min-history-days", type=int, default=180)
    parser.add_argument("--origin", action="append", help="Explicit observed origin, YYYY-MM-DD HH:00; repeatable")
    args = parser.parse_args()
    try:
        validate_recursive_features()
        history = load_development_history()
        origins = choose_origins(history, args.origins, args.spacing_days, args.min_history_days, args.origin)
        forecasts, training = evaluate(history, origins)
        horizon = metric_table(forecasts, ["Strategy", "Horizon"])
        overall = metric_table(forecasts, ["Strategy"])
        by_origin = metric_table(forecasts, ["Origin", "Strategy"])
        directory = BASE / "outputs" / ("forecast_24h_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                                       + "_" + uuid4().hex[:8])
        directory.mkdir(parents=True, exist_ok=False)
        for name, table in (("forecasts", forecasts), ("horizon_metrics", horizon),
                            ("overall_metrics", overall), ("origin_metrics", by_origin), ("training_windows", training)):
            table.to_csv(directory / f"{name}.csv", index=False)
        settings = dict(arguments=vars(args), origins=[str(origin) for origin in origins],
                        cutoff_exclusive=str(CUTOFF), features=FEATURE_COLUMNS, rf_params=RF_PARAMS,
                        target="Count - lag_1", training_labels="strictly before origin",
                        available_observations="through origin inclusive", horizon_hours=HOURS,
                        refit="once per origin", feature_validation="passed", disclaimer=DISCLAIMER,
                        python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__,
                        sklearn=sklearn.__version__)
        (directory / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
        save_figures(forecasts, horizon, overall, directory)
        report = (DISCLAIMER + "\n\nOverall metrics:\n" + overall.to_string(index=False, float_format="%.4f")
                  + "\n\nMetrics by horizon:\n" + horizon.to_string(index=False, float_format="%.4f")
                  + "\n\nPersistence repeats the origin count for all 24 hours. Seasonal naive repeats "
                  "the observed 24-hour sequence ending at the origin. All strategies use identical outcomes.\n")
        (directory / "report.txt").write_text(report, encoding="utf-8")
        print(report)
        print(f"Saved forecasts, metrics, settings, report and three figures to {directory}")
    except (OSError, ValueError, TypeError, KeyError, AssertionError) as error:
        print(f"24-hour forecasting could not complete: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
