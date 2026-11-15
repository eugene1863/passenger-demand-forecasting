"""Compare reproducible rolling one-step-ahead passenger demand forecasts.

At each timestamp, actual counts through t-1 are available. Validation history
therefore includes earlier validation observations. This is not a forecast of an
entire future week from one origin: that would require unknown intervening
counts, recursive predictions, or horizon-specific models.
"""

import json
import platform
from pathlib import Path

try:
    import joblib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import sklearn
    import xgboost
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    try:
        from sklearn.metrics import root_mean_squared_error
    except ImportError:
        # Compatibility with older sklearn; never use deprecated squared=False.
        root_mean_squared_error = None
    from xgboost import XGBRegressor
except ImportError as error:
    raise SystemExit(
        f"Missing or incompatible dependency: {error}. Install with: "
        "python -m pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib"
    ) from error

BASE = Path(__file__).resolve().parent
FEATURE_COLUMNS = [
    "hour", "day_of_week", "day_of_month", "month", "year", "is_weekend",
    "week_of_year", "lag_1", "lag_2", "lag_3", "lag_24", "lag_48", "lag_168",
    "rolling_mean_24", "rolling_mean_168", "rolling_std_24", "rolling_std_168",
    "rolling_min_24", "rolling_max_24",
]
BASELINES = {"Previous hour": "lag_1", "Previous day": "lag_24", "Previous week": "lag_168"}
ASSUMPTION = (
    "Rolling one-step-ahead: actual demand through t-1 is available at each t. "
    "Historical validation observations may enter subsequent features. This is "
    "not an unobserved future-week forecast; no recursive forecasting is implemented."
)


def load_split(path: Path) -> pd.DataFrame:
    """Reject invalid splits instead of silently sorting or imputing them."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run features.py first.")
    frame = pd.read_csv(path)
    missing = set(["Datetime", "Count"] + FEATURE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    if len(frame) < 2:
        raise ValueError(f"{path.name}: at least two observations are required.")
    dates = pd.to_datetime(frame["Datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError(f"{path.name}: timestamps must parse, be unique, and be chronological.")
    if not dates.eq(dates.dt.floor("h")).all():
        raise ValueError(f"{path.name}: timestamps must be on exact hours.")
    frame["Datetime"] = dates
    columns = ["Count"] + FEATURE_COLUMNS
    frame[columns] = frame[columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
        raise ValueError(f"{path.name}: target/features contain missing or infinite values.")
    if frame["Count"].lt(0).any():
        raise ValueError(f"{path.name}: Count must be nonnegative.")
    print(f"{path.name}: {len(frame):,} rows, {dates.iloc[0]} to {dates.iloc[-1]}")
    return frame


def score(actual, predicted) -> dict:
    """Calculate metrics on the same clipped predictions saved for inference."""
    rmse = (root_mean_squared_error(actual, predicted)
            if root_mean_squared_error is not None
            else np.sqrt(mean_squared_error(actual, predicted)))
    return {"Validation MAE": float(mean_absolute_error(actual, predicted)),
            "Validation RMSE": float(rmse), "Validation R²": float(r2_score(actual, predicted))}


def save_figures(validation, predictions, models, comparison, selected, directory):
    """Save validation diagnostics and separate tree feature importance charts."""
    sns.set_theme(style="whitegrid")

    def save(name, title, xlabel, ylabel, legend=False):
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if legend:
            plt.legend()
        plt.tight_layout()
        plt.savefig(directory / name, dpi=160)
        plt.close()

    dates = validation["Datetime"]
    # A reproducible central calendar week, independent of prediction errors.
    start = dates.iloc[len(dates) // 2].normalize()
    week = (dates >= start) & (dates < start + pd.Timedelta(days=7))
    plt.figure(figsize=(13, 5))
    plt.plot(dates[week], validation.loc[week, "Count"], label="Actual", linewidth=1.5)
    plt.plot(dates[week], predictions[selected][week], label=selected, alpha=0.8)
    save("actual_vs_predicted.png", "Central validation week: rolling one-step predictions",
         "Timestamp", "Passengers", True)
    residual = validation["Count"].to_numpy() - predictions[selected]
    plt.figure(figsize=(9, 5))
    sns.histplot(residual, bins=40, label="Actual minus predicted")
    plt.axvline(0, color="black", linestyle="--", label="Zero error")
    save("residual_distribution.png", f"Validation residuals: {selected}", "Residual (passengers)", "Hours", True)
    plt.figure(figsize=(13, 5))
    plt.plot(dates, residual, label="Actual minus predicted", linewidth=0.7)
    plt.axhline(0, color="black", linestyle="--", label="Zero error")
    save("residuals_over_time.png", f"Validation residuals over time: {selected}", "Timestamp", "Residual (passengers)", True)
    for name, model in models.items():
        importance = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values()
        plt.figure(figsize=(10, 8))
        importance.plot.barh(label="Impurity / gain importance")
        save(f"{name.lower().replace(' ', '_')}_importance.png", f"{name}: training feature importance",
             "Relative importance", "Feature", True)
    plt.figure(figsize=(10, 5))
    plt.barh(comparison["Model"], comparison["Validation MAE"], label="Validation MAE")
    plt.gca().invert_yaxis()
    save("validation_mae_comparison.png", "All approaches ranked by validation MAE", "MAE (passengers)", "Approach", True)


def main() -> int:
    """Train on training data, select on validation data, and persist artifacts."""
    print(f"scikit-learn {sklearn.__version__}; XGBoost {xgboost.__version__}")
    print(ASSUMPTION)
    try:
        train, validation, test = [load_split(BASE / "data" / f"{name}_features.csv")
                                   for name in ("train", "val", "test")]
        if not (train["Datetime"].max() < validation["Datetime"].min()
                and validation["Datetime"].max() < test["Datetime"].min()):
            raise ValueError("Train, validation, and test date ranges overlap or are out of order.")
        del test  # Schema/date validation only: no test predictions or metrics.
        models = {
            "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=15,
                min_samples_leaf=2, random_state=42, n_jobs=-1),
            "XGBoost": XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror",
                random_state=42, n_jobs=-1),
        }
        predictions = {name: np.clip(validation[column].to_numpy(), 0, None)
                       for name, column in BASELINES.items()}
        for name, model in models.items():
            print(f"Training {name} on training observations only...")
            model.fit(train[FEATURE_COLUMNS], train["Count"])
            predictions[name] = np.clip(model.predict(validation[FEATURE_COLUMNS]), 0, None)
        comparison = pd.DataFrame([
            {"Model": name, **score(validation["Count"], prediction)}
            for name, prediction in predictions.items()
        ]).sort_values("Validation MAE", kind="stable").reset_index(drop=True)
        selected = comparison.iloc[0]["Model"]
        strongest_baseline = next(name for name in comparison["Model"] if name in BASELINES)
        print("\n" + comparison.to_string(index=False))
        print(f"\nStrongest baseline: {strongest_baseline}. Selected by validation MAE: {selected}.")
        print("Baselines are cheap and interpretable; forests average independent trees; "
              "boosting adds trees to correct errors. Extra training and inference costs "
              "are justified only by useful validation improvement. Test data remains unevaluated.")
        print("Predictions are clipped to zero because passenger demand cannot be negative; "
              "all reported metrics use these final predictions. Counts are not rounded.")
        print("Tree importance describes model reliance, not causality; correlated lags can share importance.")
        output, model_dir = BASE / "outputs", BASE / "models"
        figures = output / "figures"
        for directory in (output, model_dir, figures):
            directory.mkdir(parents=True, exist_ok=True)
        versions = {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                    "xgboost": xgboost.__version__, "pandas": pd.__version__, "numpy": np.__version__}
        metadata = {"feature_columns": FEATURE_COLUMNS, "selected_approach": selected,
                    "strongest_baseline": strongest_baseline, "selection_metric": "Validation MAE",
                    "baseline_column": BASELINES.get(selected), "clip_predictions_min": 0,
                    "assumption": ASSUMPTION, "versions": versions, "random_state": 42,
                    "training_start": str(train["Datetime"].min()),
                    "training_end": str(train["Datetime"].max()),
                    "validation_start": str(validation["Datetime"].min()),
                    "validation_end": str(validation["Datetime"].max())}
        for name, model in models.items():
            filename = "random_forest.joblib" if name == "Random Forest" else "xgboost.joblib"
            # Bundle preserves the inference schema and postprocessing alongside estimator.
            joblib.dump({"model": model, "metadata": {**metadata, "model_name": name}}, model_dir / filename)
        (output / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        comparison.to_csv(output / "model_comparison.csv", index=False)
        forecast = validation[["Datetime", "Count"]].copy()
        for name, prediction in predictions.items():
            forecast[name] = prediction
        forecast["selected_prediction"] = predictions[selected]
        forecast["selected_residual"] = forecast["Count"] - forecast["selected_prediction"]
        forecast.to_csv(output / "validation_predictions.csv", index=False)
        save_figures(validation, predictions, models, comparison, selected, figures)
        print(f"Saved models to {model_dir} and validation artifacts to {output}.")
    except (OSError, ValueError, TypeError, pd.errors.ParserError, xgboost.core.XGBoostError) as error:
        print(f"Training could not complete: {error}")
        print("Check input schemas and installed versions. Install compatible dependencies with "
              "python -m pip install --upgrade pandas numpy matplotlib seaborn scikit-learn xgboost joblib")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
