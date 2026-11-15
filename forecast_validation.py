"""Rolling-origin 24-hour validation on retrospective pre-test development data.

Run python forecast_validation.py --help. No model selection or tuning is done.
Weekly origins are the default; --stride-days 1 evaluates daily origins.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import platform
import sys
from time import perf_counter
from uuid import uuid4

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.ensemble import RandomForestRegressor

from backtesting import CUTOFF, RF_PARAMS, metrics
from features import FEATURE_COLUMNS, add_features
from forecast_24h import (
    BASE, HOURS, STRATEGIES, DISCLAIMER, choose_origins,
    load_development_history, recursive_forecast, validate_recursive_features,
)

LOGGER = logging.getLogger(__name__)
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DEPENDENCE = (
    "Origin scores are dependent: expanding training sets overlap and demand is serially "
    "correlated. Daily 24-hour windows abut; weekly windows are separated, but neither "
    "provides independent samples. If forecast windows overlap, repeated targets are "
    "weighted once per forecast case in pooled metrics. No IID confidence intervals or "
    "significance claims are made. Lower development error does not prove future "
    "generalization; fully independent evaluation requires new unseen chronological data."
)


def parse_hour(value: str) -> pd.Timestamp:
    """Parse a timezone-naive, exact-hour timestamp without accepting NaT."""
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is not None or stamp != stamp.floor("h"):
        raise ValueError("Start/end must be timezone-naive exact clock hours.")
    return stamp


def generate_origins(history: pd.DataFrame, stride_days: int, min_history_days: int,
                     start: str | None, end: str | None) -> tuple[list[pd.Timestamp], pd.DataFrame]:
    """Anchor a regular grid at start; audit and skip incomplete candidate windows."""
    if stride_days < 1 or min_history_days < 1:
        raise ValueError("Stride and minimum history days must be positive integers.")
    earliest = history.index.min() + pd.Timedelta(days=min_history_days)
    latest = min(history.index.max() - pd.Timedelta(hours=HOURS),
                 CUTOFF - pd.Timedelta(hours=HOURS + 1))
    first = parse_hour(start) if start else earliest.ceil("D")
    last = parse_hour(end) if end else latest
    if first < earliest or last > latest or first > last:
        raise ValueError(f"Origin bounds must satisfy {earliest} <= start <= end <= {latest}.")
    origins, audit = [], []
    for origin in pd.date_range(first, last, freq=pd.Timedelta(days=stride_days)):
        reason = ""
        try:
            choose_origins(history, 1, stride_days, min_history_days, [str(origin)])
        except ValueError as error:
            reason = str(error)
        if not reason:
            origins.append(origin)
        audit.append(dict(Origin=origin, Included=not bool(reason), Exclusion_Reason=reason))
    if not origins:
        raise ValueError("No candidate origins have complete context and evaluation outcomes.")
    return origins, pd.DataFrame(audit)


def run_validation(history: pd.DataFrame, origins: list[pd.Timestamp],
                   directory: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cache causal training features, refit once per origin, then recurse in isolation.

    Labels strictly precede the origin, preserving forecast_24h.py's convention.
    The cached actual-derived table is NEVER supplied to recursive_forecast.
    """
    cached = add_features(history).dropna(subset=["Count", *FEATURE_COLUMNS])
    delta = cached.Count - cached.lag_1
    forecasts, windows = [], []
    for number, origin in enumerate(origins, 1):
        tick = perf_counter()
        train = cached.loc[cached.index < origin]
        if len(train) < 2:
            raise ValueError(f"Too few complete training rows at {origin}.")
        # Verify the cached prefix equals feature engineering with future rows absent.
        expected = add_features(history.loc[history.index < origin]).reindex(train.index)
        np.testing.assert_allclose(train[FEATURE_COLUMNS].to_numpy(dtype=float),
                                   expected[FEATURE_COLUMNS].to_numpy(dtype=float),
                                   rtol=1e-10, atol=1e-10)
        LOGGER.info("Origin %d/%d %s: fitting %s rows through %s", number, len(origins),
                    origin, f"{len(train):,}", train.index.max())
        model = RandomForestRegressor(**RF_PARAMS)
        model.fit(train[FEATURE_COLUMNS], delta.loc[train.index])
        observed = history.loc[:origin, ["Count"]].copy()
        rf = recursive_forecast(model, observed)
        for strategy in STRATEGIES:
            frame = rf.copy()
            if strategy != STRATEGIES[0]:
                frame[["Predicted_Delta", "Previous_Working_Count"]] = np.nan
                frame["Predicted_Count"] = (float(observed.Count.iloc[-1])
                    if strategy == STRATEGIES[1] else observed.Count.iloc[-HOURS:].to_numpy())
            # Actuals are attached only AFTER the entire horizon has been generated.
            frame["Origin"] = origin
            frame["Strategy"] = strategy
            frame["Actual_Count"] = history.Count.reindex(pd.DatetimeIndex(frame.Datetime)).to_numpy()
            frame["Error"] = frame.Predicted_Count - frame.Actual_Count
            frame["Absolute_Error"] = frame.Error.abs()
            if not np.isfinite(frame[["Actual_Count", "Predicted_Count"]].to_numpy()).all():
                raise ValueError(f"Nonfinite forecasts/outcomes at {origin}.")
            forecasts.append(frame)
        windows.append(dict(Origin=origin, Training_Start=train.index.min(),
                            Training_End=train.index.max(), Training_Observations=len(train),
                            Elapsed_Seconds=perf_counter() - tick))
        # Save completed work incrementally; a long interrupted run remains auditable.
        pd.concat(forecasts, ignore_index=True).to_csv(directory / "validation_predictions.csv", index=False)
        pd.DataFrame(windows).to_csv(directory / "validation_training_windows.csv", index=False)
        LOGGER.info("Completed origin %d/%d in %.1fs", number, len(origins), perf_counter() - tick)
    result = pd.concat(forecasts, ignore_index=True)
    expected_cases = len(origins) * HOURS
    for _, part in result.groupby("Strategy"):
        if len(part) != expected_cases or part.duplicated(["Origin", "Datetime"]).any():
            raise AssertionError("Strategies do not have identical complete forecast cases.")
    return result, pd.DataFrame(windows)


def grouped_metrics(forecasts: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    """Pool individual errors; WAPE is percent, bias is predicted minus actual."""
    rows = []
    grouper = groups[0] if len(groups) == 1 else groups
    for key, part in forecasts.groupby(grouper, sort=True, observed=True):
        keys = key if isinstance(key, tuple) else (key,)
        rows.append(dict(zip(groups, keys), Forecasts=len(part), Origins=part.Origin.nunique(),
                         Mean_Actual=float(part.Actual_Count.mean()),
                         **metrics(part.Actual_Count.to_numpy(), part.Predicted_Count.to_numpy())))
    return pd.DataFrame(rows)


def diagnostic_tables(forecasts: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, float]]:
    """Add target-time calendar strata and retrospective actual-demand quartile bins."""
    forecasts["Weekday"] = pd.Categorical(forecasts.Datetime.dt.day_name(), WEEKDAYS, ordered=True)
    forecasts["Day_Type"] = np.where(forecasts.Datetime.dt.dayofweek >= 5, "Weekend", "Weekday")
    # Unique outcomes avoid weighting duplicated targets when deriving thresholds.
    actuals = forecasts.drop_duplicates("Datetime").Actual_Count
    q25, q75 = (float(x) for x in actuals.quantile([.25, .75]))
    forecasts["Demand_Level"] = np.where(forecasts.Actual_Count > q75, "High",
                                          np.where(forecasts.Actual_Count <= q25, "Low", "Middle"))
    tables = {name: grouped_metrics(forecasts, groups) for name, groups in (
        ("overall_summary", ["Strategy"]), ("metrics_by_horizon", ["Strategy", "Horizon"]),
        ("metrics_by_origin", ["Origin", "Strategy"]), ("metrics_by_weekday", ["Strategy", "Weekday"]),
        ("metrics_by_day_type", ["Strategy", "Day_Type"]),
        ("metrics_by_demand", ["Strategy", "Demand_Level"]))}
    origin = tables["metrics_by_origin"].pivot(index="Origin", columns="Strategy", values="MAE")
    comparison = origin.rename(columns={STRATEGIES[0]: "RF_MAE", STRATEGIES[1]: "Persistence_MAE",
                                        STRATEGIES[2]: "Seasonal_MAE"}).reset_index()
    comparison["RF_minus_Seasonal_MAE"] = comparison.RF_MAE - comparison.Seasonal_MAE
    comparison["RF_Beats_Seasonal"] = comparison.RF_MAE < comparison.Seasonal_MAE
    comparison["RF_Ties_Seasonal"] = np.isclose(comparison.RF_MAE, comparison.Seasonal_MAE,
                                                rtol=0, atol=1e-10)
    comparison["RF_Beats_Persistence"] = comparison.RF_MAE < comparison.Persistence_MAE
    tables["baseline_comparisons"] = comparison
    return tables, {"low_max_inclusive": q25, "high_min_exclusive": q75}


def save_figures(forecasts: pd.DataFrame, tables: dict[str, pd.DataFrame], directory: Path) -> None:
    """Save six plots; choose the example by chronological middle, never by score."""
    sns.set_theme(style="whitegrid")

    def save(fig: plt.Figure, name: str) -> None:
        """Write and close one figure."""
        fig.tight_layout()
        fig.savefig(directory / f"validation_{name}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    summary = tables["overall_summary"]
    ax.barh(summary.Strategy, summary.MAE)
    ax.set(title="Retrospective development: pooled forecast MAE", xlabel="MAE (passengers)")
    save(fig, "overall_mae")
    for name, table, x, y, label in (
        ("horizon_mae", tables["metrics_by_horizon"], "Horizon", "MAE", "Hours after origin"),
        ("origin_mae", tables["metrics_by_origin"], "Origin", "MAE", "Forecast origin"),
        ("horizon_bias", tables["metrics_by_horizon"], "Horizon", "MBE", "Hours after origin")):
        fig, ax = plt.subplots(figsize=(12, 5))
        for strategy, part in table.groupby("Strategy", sort=False):
            ax.plot(part[x], part[y], label=strategy)
        if y == "MBE":
            ax.axhline(0, color="black", linestyle="--")
        ax.set(title=f"Retrospective development: {name.replace('_', ' ')}", xlabel=label,
               ylabel="Bias: predicted - actual" if y == "MBE" else "MAE (passengers)")
        ax.legend()
        if x == "Origin":
            fig.autofmt_xdate()
        else:
            ax.set_xticks([1, 6, 12, 18, 24])
        save(fig, name)
    origins = sorted(forecasts.Origin.unique())
    origin = origins[len(origins) // 2]
    example = forecasts.loc[forecasts.Origin.eq(origin)]
    fig, ax = plt.subplots(figsize=(12, 5))
    actual = example.loc[example.Strategy.eq(STRATEGIES[0])]
    ax.plot(actual.Datetime, actual.Actual_Count, color="black", linewidth=2, label="Actual")
    for strategy, part in example.groupby("Strategy", sort=False):
        ax.plot(part.Datetime, part.Predicted_Count, label=strategy)
    ax.set(title=f"Chronological middle development origin: {pd.Timestamp(origin)}", ylabel="Passengers")
    ax.legend()
    fig.autofmt_xdate()
    save(fig, "actual_vs_predicted")
    fig, ax = plt.subplots(figsize=(13, 5))
    weekday = tables["metrics_by_weekday"].pivot(index="Weekday", columns="Strategy", values="MAE")
    weekday.reindex(WEEKDAYS).plot.bar(ax=ax, rot=20)
    ax.set(title="Development weekday MAE (blank = unsampled)", ylabel="MAE (passengers)", xlabel="Target weekday")
    save(fig, "weekday_mae")


def make_report(forecasts: pd.DataFrame, tables: dict[str, pd.DataFrame],
                thresholds: dict[str, float]) -> str:
    """Describe wins, demand, horizons, bias and coverage without inference claims."""
    rf = forecasts.loc[forecasts.Strategy.eq(STRATEGIES[0])]
    comparison = tables["baseline_comparisons"]
    wins = int(comparison.RF_Beats_Seasonal.sum())
    ties = int(comparison.RF_Ties_Seasonal.sum())
    early = rf.loc[rf.Horizon <= 6].Absolute_Error.mean()
    late = rf.loc[rf.Horizon >= 19].Absolute_Error.mean()
    horizon = tables["metrics_by_horizon"]
    slope = np.polyfit(horizon.loc[horizon.Strategy.eq(STRATEGIES[0]), "Horizon"],
                       horizon.loc[horizon.Strategy.eq(STRATEGIES[0]), "MAE"], 1)[0]
    sampled = [day for day in WEEKDAYS if day in rf.Datetime.dt.day_name().unique()]
    lines = [DISCLAIMER, DEPENDENCE,
             f"Origins: {rf.Origin.nunique()}; forecasts per strategy: {len(rf)}; "
             f"all strategy rows: {len(forecasts)}; unique target timestamps: {rf.Datetime.nunique()}.",
             "Overall pooled metrics (MBE > 0 overprediction; MBE < 0 underprediction; WAPE in %):",
             tables["overall_summary"].to_string(index=False, float_format="%.4f"),
             f"RF beats seasonal naive at {wins}/{len(comparison)} origins ({100*wins/len(comparison):.2f}%); "
             f"ties within 1e-10: {ties}. Wins use strict lower MAE; inspect all origin differences.",
             f"RF early horizons 1-6 MAE: {early:.4f}; late horizons 19-24 MAE: {late:.4f}; "
             f"descriptive horizon-MAE linear slope: {slope:.4f} passengers/hour. "
             "A slope is not evidence of monotonic error growth or statistical significance.",
             f"Actual-demand diagnostic thresholds: {thresholds}. Low <= Q25, High > Q75, otherwise Middle. "
             "Thresholds use unique evaluated actuals after prediction, never model features or tuning. "
             "MAE scales with demand; inspect WAPE, bias and sample counts alongside MAE.",
             "Demand diagnostics:", tables["metrics_by_demand"].to_string(index=False, float_format="%.4f"),
             "Weekday/weekend diagnostics:", tables["metrics_by_day_type"].to_string(index=False, float_format="%.4f"),
             f"Sampled target weekdays: {', '.join(sampled)}.",
             "Weekly origins have fixed weekday composition and do not cover the full week. "
             "Missing weekdays/day types cannot be compared. Use --stride-days 1 for broader coverage; "
             "calendar mix, hour mix, time trend and training size confound group comparisons.",
             "Training labels strictly precede origin, matching forecast_24h.py; actual context includes origin. "
             "Forests refit once per origin; no saved models are loaded or overwritten. "
             "No test-period counts are used for fitting, validation, selection or tuning."]
    return "\n\n".join(lines) + "\n"


def source_hash(path: Path) -> str:
    """Fingerprint source code for reproducibility without reading test data."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    """Validate, evaluate, and save a uniquely named, auditable experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stride-days", type=int, choices=(1, 7), default=7)
    parser.add_argument("--min-history-days", type=int, default=180)
    parser.add_argument("--start", help="Inclusive first origin (YYYY-MM-DD HH:00); default earliest eligible midnight")
    parser.add_argument("--end", help="Inclusive upper bound for origins; all 24 targets must be pre-test")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    directory = None
    try:
        validate_recursive_features()
        history = load_development_history()
        origins, audit = generate_origins(history, args.stride_days, args.min_history_days, args.start, args.end)
        directory = BASE / "outputs" / ("forecast_validation_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                                        + "_" + uuid4().hex[:8])
        directory.mkdir(parents=True, exist_ok=False)
        handler = logging.FileHandler(directory / "validation_progress.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        LOGGER.addHandler(handler)
        settings = dict(status="running", arguments=vars(args), cutoff_exclusive=str(CUTOFF),
                        origins=[str(x) for x in origins], candidate_origins=len(audit),
                        excluded_origins=int((~audit.Included).sum()), features=FEATURE_COLUMNS,
                        rf_params=RF_PARAMS, target="Count - lag_1", training_labels="strictly before origin",
                        observed_context="through origin inclusive", horizon_hours=HOURS,
                        refit="once per origin; unchanged RF configuration", cache="in-memory causal training features only",
                        python=platform.python_version(), executable=sys.executable, platform=platform.platform(),
                        packages={"pandas": pd.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__,
                                  "matplotlib": matplotlib.__version__, "seaborn": sns.__version__},
                        source_sha256={name: source_hash(BASE / name) for name in
                                       ("features.py", "backtesting.py", "forecast_24h.py", "forecast_validation.py")},
                        disclaimer=DISCLAIMER, dependence=DEPENDENCE, feature_validation="passed")
        settings_path = directory / "validation_settings.json"
        settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
        audit.to_csv(directory / "validation_origin_audit.csv", index=False)
        LOGGER.info("Evaluating %d origins; %d excluded. Output: %s", len(origins), (~audit.Included).sum(), directory)
        forecasts, _ = run_validation(history, origins, directory)
        tables, thresholds = diagnostic_tables(forecasts)
        for name, table in tables.items():
            table.to_csv(directory / f"validation_{name}.csv", index=False)
        forecasts.to_csv(directory / "validation_predictions.csv", index=False)
        save_figures(forecasts, tables, directory)
        report = make_report(forecasts, tables, thresholds)
        (directory / "validation_report.txt").write_text(report, encoding="utf-8")
        settings.update(status="complete", demand_thresholds=thresholds, origins_evaluated=len(origins),
                        forecasts_per_strategy=len(origins) * HOURS,
                        unique_target_timestamps=forecasts.Datetime.nunique(),
                        feature_validation="recursive parity/isolation and cached-prefix parity passed")
        settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
        print(report, flush=True)
        LOGGER.info("Complete. Saved CSV metrics, predictions, settings, report and six figures to %s", directory)
    except (OSError, ValueError, TypeError, KeyError, AssertionError) as error:
        LOGGER.error("Validation could not complete: %s", error)
        if directory is not None and directory.exists():
            LOGGER.error("Partial artifacts, if present, are incomplete: %s", directory)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
