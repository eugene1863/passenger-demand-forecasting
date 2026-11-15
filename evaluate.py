"""Evaluate the saved Random Forest once on held-out rolling one-step test data.

Run: python evaluate.py. Only this project's trusted local joblib is loaded.
No fitting, tuning, feature changes, or model selection occurs here.
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = Path(__file__).resolve().parent
BASELINES = {"Previous hour": "lag_1", "Previous day": "lag_24", "Previous week": "lag_168"}
ASSUMPTION = (
    "Rolling one-step-ahead: actual passenger counts through t-1 become available "
    "at each prediction timestamp. This is not a fixed-origin multi-day or future-week forecast."
)
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def load_model() -> tuple[RandomForestRegressor, list[str], dict]:
    """Read trusted local training artifacts and cross-check their inference schema."""
    bundle = joblib.load(BASE / "models" / "random_forest.joblib")
    metadata = bundle.get("metadata", {}) if isinstance(bundle, dict) else {}
    model = bundle["model"] if isinstance(bundle, dict) else bundle
    metadata_path = BASE / "outputs" / "model_metadata.json"
    external = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    for key in ("feature_columns", "selected_approach", "validation_start", "validation_end"):
        if key in metadata and key in external and metadata[key] != external[key]:
            raise ValueError(f"Conflicting training metadata: {key}.")
    metadata = {**external, **metadata}
    if metadata.get("selected_approach", "Random Forest") != "Random Forest":
        raise ValueError("Training metadata does not identify Random Forest as the selected approach.")
    if not isinstance(model, RandomForestRegressor):
        raise ValueError("Expected a fitted RandomForestRegressor in the local artifact.")
    fitted_names = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else None
    columns = metadata.get("feature_columns", fitted_names)
    if not isinstance(columns, list) or not columns or not all(isinstance(c, str) for c in columns):
        raise ValueError("No valid saved feature order; evaluation cannot guess the schema.")
    if len(set(columns)) != len(columns) or any(c.lower() in {"datetime", "count", "id"} or c.lower().endswith("_id") for c in columns):
        raise ValueError("Saved schema contains duplicate features, identifiers, or the target.")
    if fitted_names is not None and columns != fitted_names:
        raise ValueError("Metadata feature order differs from fitted feature_names_in_.")
    if getattr(model, "n_features_in_", None) != len(columns) or not hasattr(model, "estimators_"):
        raise ValueError("Model is unfitted or incompatible with its feature schema.")
    return model, columns, metadata


def load_data(path: Path, columns: list[str]) -> pd.DataFrame:
    """Reject invalid data without sorting, dropping, filling, or changing source files."""
    frame = pd.read_csv(path)
    required = list(dict.fromkeys(["Count", *columns, *BASELINES.values()]))
    missing = set(["Datetime", *required]) - set(frame.columns)
    if frame.empty or missing:
        raise ValueError(f"{path.name}: empty dataset or missing columns {sorted(missing)}.")
    dates = pd.to_datetime(frame["Datetime"], format="%Y-%m-%d %H:%M:%S", errors="raise")
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError(f"{path.name}: timestamps must be valid, unique, and chronological.")
    if not dates.eq(dates.dt.floor("h")).all():
        raise ValueError(f"{path.name}: timestamps must fall on exact hours.")
    frame["Datetime"] = dates
    frame[required] = frame[required].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame[required].to_numpy(dtype=float)).all():
        raise ValueError(f"{path.name}: missing or nonfinite target/features.")
    if frame["Count"].lt(0).any() or frame[list(BASELINES.values())].lt(0).any().any():
        raise ValueError(f"{path.name}: negative passenger counts or historical baselines.")
    return frame


def validate_history(test: pd.DataFrame, validation: pd.DataFrame, columns: list[str], metadata: dict) -> None:
    """Audit test lags/rolling values against earlier observations on an hourly grid."""
    if test["Datetime"].iloc[0] <= validation["Datetime"].iloc[-1]:
        raise ValueError("Test dates must begin strictly after validation dates.")
    for key, actual in (("validation_start", validation["Datetime"].iloc[0]),
                        ("validation_end", validation["Datetime"].iloc[-1])):
        if key in metadata and pd.Timestamp(metadata[key]) != actual:
            raise ValueError(f"Validation CSV disagrees with saved {key}.")
    # Validation supplies the pre-test context; current test counts are shifted
    # before any rolling calculation, so they cannot enter their own predictors.
    history = pd.concat([validation, test]).set_index("Datetime")["Count"]
    history = history.reindex(pd.date_range(history.index.min(), history.index.max(), freq="h"))
    dates = pd.DatetimeIndex(test["Datetime"])
    for column in columns:
        expected = None
        if column.startswith("lag_"):
            lag = int(column.split("_")[-1])
            if lag <= 0:
                raise ValueError("Historical lags must be positive.")
            expected = history.shift(lag)
        elif column.startswith("rolling_"):
            _, operation, window = column.split("_")
            rolling = history.shift(1).rolling(int(window), min_periods=int(window))
            methods = {"mean": rolling.mean, "std": rolling.std, "min": rolling.min, "max": rolling.max}
            if operation not in methods:
                raise ValueError(f"Unknown rolling operation: {column}")
            expected = methods[operation]()
        if expected is not None:
            values = expected.reindex(dates).to_numpy()
            if not np.isfinite(values).all() or not np.allclose(values, test[column], rtol=1e-7, atol=1e-7):
                raise ValueError(f"Cannot verify historical-only feature {column} from validation/test observations.")
    print("Verified test lag and rolling features against strictly earlier actual observations.")


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """MBE is predicted minus actual; WAPE is undefined when total actual is zero."""
    total = float(np.sum(actual))
    return {"MAE": float(mean_absolute_error(actual, predicted)),
            "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
            "R2": float(r2_score(actual, predicted, force_finite=False)) if len(actual) > 1 and np.var(actual) > 0 else np.nan,
            "MBE": float(np.mean(predicted - actual)),
            "WAPE (%)": float(100 * np.sum(np.abs(actual - predicted)) / total) if total > 0 else np.nan}


def segment_errors(predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Describe errors by calendar groups and test-only actual-demand quartiles."""
    frame = predictions.copy()
    calendar = frame["Datetime"].dt
    frame["hour"] = calendar.hour
    frame["weekday"] = pd.Categorical(calendar.day_name(), categories=WEEKDAYS, ordered=True)
    frame["month"] = calendar.month
    frame["weekend"] = np.where(calendar.dayofweek >= 5, "Weekend", "Weekday")
    # Duplicate quantile edges are dropped; ties are never split arbitrarily.
    quartiles, edges = pd.qcut(frame["Actual Count"], 4, duplicates="drop", retbins=True)
    if len(edges) < 2:
        frame["demand_quartile"] = "All demand (constant actual)"
    else:
        frame["demand_quartile"] = quartiles.cat.rename_categories(
            [f"Q{i + 1}: {interval}" for i, interval in enumerate(quartiles.cat.categories)])
    print("Actual-demand quantile edges (descriptive only):", edges)
    if len(edges) != 5:
        print("Tied demand values prevent four distinct quartiles; reporting available groups.")
    return {name: frame.groupby(name, observed=True, sort=True).agg(
        Observations=("Absolute Error", "size"), MAE=("Absolute Error", "mean"),
        Mean_Actual=("Actual Count", "mean")).reset_index()
        for name in ("hour", "weekday", "month", "weekend", "demand_quartile")}


def save_figures(predictions: pd.DataFrame, segments: dict, comparison: pd.DataFrame, directory: Path) -> None:
    """Save nine test diagnostics without interactive GUI dependencies."""
    directory.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    def finish(figure, axis, name, title, xlabel, ylabel):
        axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
        axis.legend()
        figure.tight_layout()
        figure.savefig(directory / f"test_{name}.png", dpi=160)
        plt.close(figure)

    first_week = predictions["Datetime"] < predictions["Datetime"].iloc[0] + pd.Timedelta(days=7)
    for name, frame, title in (("actual_vs_predicted", predictions, "Complete test period"),
                               ("first_seven_days", predictions.loc[first_week], "First seven test days")):
        fig, ax = plt.subplots(figsize=(13, 5))
        for column in ("Actual Count", "Predicted Count"):
            ax.plot(frame["Datetime"], frame[column], label=column, linewidth=0.9, alpha=0.85)
        finish(fig, ax, name, f"{title}: rolling one-step Random Forest", "Timestamp", "Passengers per hour")
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(predictions["Actual Count"], predictions["Predicted Count"], alpha=0.3, s=12, label="Test observations")
    limit = predictions[["Actual Count", "Predicted Count"]].to_numpy().max()
    ax.plot([0, limit], [0, limit], "k--", label="Perfect prediction")
    finish(fig, ax, "scatter", "Random Forest: test actual versus predicted", "Actual passengers", "Predicted passengers")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(predictions["Residual"], bins=40, ax=ax, label="Actual minus predicted")
    ax.axvline(0, color="black", linestyle="--", label="Zero residual")
    finish(fig, ax, "residual_histogram", "Test residual distribution", "Residual (passengers)", "Observations")
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(predictions["Datetime"], predictions["Residual"], linewidth=0.7, label="Actual minus predicted")
    ax.axhline(0, color="black", linestyle="--", label="Zero residual")
    finish(fig, ax, "residuals_over_time", "Test residuals over time", "Timestamp", "Residual (passengers)")
    for group, title in (("hour", "Hour of day"), ("weekday", "Day of week"), ("demand_quartile", "Actual-demand quartile")):
        table = segments[group]
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(table[group].astype(str), table["MAE"], label="Random Forest test MAE")
        ax.tick_params(axis="x", rotation=25 if group != "hour" else 0)
        finish(fig, ax, f"mae_by_{group}", f"Test MAE by {title.lower()}", title, "MAE (passengers)")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(comparison["Model"], comparison["MAE"], label="Test MAE")
    ax.invert_yaxis()
    finish(fig, ax, "mae_comparison", "Test MAE: selected Random Forest and historical baselines", "MAE (passengers)", "Approach")


def print_report(predictions: pd.DataFrame, comparison: pd.DataFrame, segments: dict) -> None:
    """Explain metrics, bias, worst errors, and the limits of held-out evaluation."""
    forest = comparison.loc[comparison["Model"].eq("Random Forest")].iloc[0]
    best = comparison.loc[comparison["Model"].isin(BASELINES)].sort_values("MAE", kind="stable").iloc[0]
    improvement = 100 * (best["MAE"] - forest["MAE"]) / best["MAE"] if best["MAE"] > 0 else np.nan
    print(f"\nTest observations: {len(predictions):,}")
    print(f"Test range: {predictions['Datetime'].iloc[0]} to {predictions['Datetime'].iloc[-1]}")
    print("\n" + comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("MAE: average absolute error (passengers). RMSE: squared-error-sensitive error (passengers).")
    print("R2: dimensionless fit relative to the test mean; negative values are possible; undefined for constant targets.")
    print("MBE: mean predicted minus actual (passengers); positive means overprediction, negative underprediction.")
    print("WAPE: total absolute error / total actual demand (%); undefined if total actual is zero. MAPE is not used.")
    print(f"\nStrongest historical test baseline: {best['Model']}, MAE={best['MAE']:.4f} passengers.")
    print(f"Relative Random Forest MAE improvement: {improvement:.2f}%" if np.isfinite(improvement) else "Relative MAE improvement undefined: baseline MAE is zero.")
    verdict = "beats" if forest["MAE"] < best["MAE"] else "ties" if forest["MAE"] == best["MAE"] else "underperforms"
    print(f"Selected Random Forest {verdict} the strongest test baseline on MAE. Candidate remains unchanged.")
    residual = predictions["Residual"]
    print(f"\nResidual mean={residual.mean():.4f}, median={residual.median():.4f}, sample std={residual.std(ddof=1):.4f} passengers.")
    print(f"Mean prediction bias (MBE)={forest['MBE']:.4f} passengers.")
    print("Positive residuals indicate underprediction; negative residuals indicate overprediction.")
    direction = "underprediction" if residual.mean() > 0 else "overprediction" if residual.mean() < 0 else "zero mean bias"
    print(f"Observed average error direction: {direction}; this is descriptive, not a significance claim.")
    print(f"Underpredicted hours: {(residual > 0).mean():.2%}; overpredicted hours: {(residual < 0).mean():.2%}.")
    print("\nLargest absolute errors:\n" + predictions.nlargest(10, "Absolute Error").to_string(index=False))
    for group, table in segments.items():
        best_row, worst_row = table.loc[table["MAE"].idxmin()], table.loc[table["MAE"].idxmax()]
        print(f"{group}: lowest MAE {best_row[group]} ({best_row['MAE']:.4f}); highest MAE {worst_row[group]} ({worst_row['MAE']:.4f} passengers).")
    print("Segment comparisons are descriptive; observation counts and demand levels differ between groups.")
    print("Predictions clipped at zero; continuous values used for every metric. No fitting or tuning performed.")
    print(ASSUMPTION)


def main() -> int:
    """Validate artifacts, predict once, and save the independent test evaluation."""
    try:
        print(ASSUMPTION)
        model, columns, metadata = load_model()
        test = load_data(BASE / "data" / "test_features.csv", columns)
        validation = load_data(BASE / "data" / "val_features.csv", columns)
        validate_history(test, validation, columns, metadata)
        actual = test["Count"].to_numpy(dtype=float)
        raw = np.asarray(model.predict(test[columns]), dtype=float)
        if raw.shape != actual.shape or not np.isfinite(raw).all():
            raise ValueError("Model returned invalid predictions.")
        predicted = np.clip(raw, 0, None)
        predictions = pd.DataFrame({"Datetime": test["Datetime"], "Actual Count": actual,
                                    "Predicted Count": predicted, "Residual": actual - predicted,
                                    "Absolute Error": np.abs(actual - predicted)})
        comparison = pd.DataFrame([
            {"Model": "Random Forest", **metrics(actual, predicted)},
            *[{"Model": name, **metrics(actual, np.clip(test[column].to_numpy(dtype=float), 0, None))}
              for name, column in BASELINES.items()]])
        segments = segment_errors(predictions)
        output = BASE / "outputs"
        output.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(output / "test_predictions.csv", index=False)
        comparison.iloc[:1].to_csv(output / "test_metrics.csv", index=False)
        comparison.to_csv(output / "test_baseline_comparison.csv", index=False)
        for group, table in segments.items():
            table.to_csv(output / f"test_error_by_{group}.csv", index=False)
        save_figures(predictions, segments, comparison, output / "figures")
        print_report(predictions, comparison, segments)
        print(f"Test evaluation artifacts saved to {output}")
    except (OSError, ValueError, TypeError, KeyError, AttributeError, ImportError) as error:
        print(f"Test evaluation could not complete: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
