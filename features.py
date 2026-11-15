"""Build historical-only features for one-step-ahead hourly demand forecasting.

Run ``python features.py``. At timestamp t, observed counts through t-1 are
available, including earlier observations within validation/test periods. This
is rolling one-step evaluation. A forecast made for several future hours at once
must recursively use predictions or use horizon-specific models; future actual
counts must never be supplied as if already observed.
No scalers, imputers, or other learned transformations are fitted here.
"""

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIRECTORY = Path(__file__).resolve().parent
DATA_DIRECTORY = BASE_DIRECTORY / "data"
DATETIME_FORMAT = "%d-%m-%Y %H:%M"
LAGS = (1, 2, 3, 24, 48, 168)
CALENDAR_COLUMNS = [
    "hour", "day_of_week", "day_of_month", "month", "year", "is_weekend",
    "week_of_year",
]
ROLLING_COLUMNS = [
    "rolling_mean_24", "rolling_mean_168", "rolling_std_24",
    "rolling_std_168", "rolling_min_24", "rolling_max_24",
]
FEATURE_COLUMNS = CALENDAR_COLUMNS + [f"lag_{lag}" for lag in LAGS] + ROLLING_COLUMNS


def validate_time_index(data: pd.DataFrame, require_hourly: bool = True) -> None:
    """Reject empty, unordered, duplicate, or off-grid timestamp indices."""
    index = data.index
    if not isinstance(index, pd.DatetimeIndex) or data.empty or index.hasnans:
        raise ValueError("Expected a nonempty DatetimeIndex without missing timestamps.")
    if not index.is_monotonic_increasing or not index.is_unique:
        raise ValueError("Timestamps must be sorted and unique.")
    if not index.equals(index.floor("h")):
        raise ValueError("Datetime values must fall on exact clock hours (HH:00).")
    if require_hourly and not index.to_series().diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
        raise ValueError("The time index must contain every consecutive hour.")


def load_dataset(path: Path) -> pd.DataFrame:
    """Validate source observations and insert missing hours with NaN targets.

    Source IDs are identifiers, not predictors. Only Datetime and Count are kept.
    Blank counts are allowed; malformed, infinite, or negative counts are rejected.
    """
    data = pd.read_csv(path)
    missing = {"ID", "Datetime", "Count"}.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if data.empty:
        raise ValueError("The input dataset contains no observations.")
    dates = pd.to_datetime(data["Datetime"], format=DATETIME_FORMAT, errors="coerce")
    if dates.isna().any():
        raise ValueError("Missing or invalid Datetime; expected DD-MM-YYYY HH:MM.")
    if dates.duplicated().any():
        examples = dates[dates.duplicated(keep=False)].head().astype(str).tolist()
        raise ValueError(f"Duplicate timestamps are not allowed. Examples: {examples}")
    counts = pd.to_numeric(data["Count"], errors="coerce")
    invalid = data["Count"].notna() & counts.isna()
    if invalid.any() or (counts.notna() & ~np.isfinite(counts)).any():
        raise ValueError("Count contains nonnumeric or infinite values.")
    if counts.lt(0).any():
        raise ValueError("Count contains negative passenger counts.")
    if not counts.notna().any():
        raise ValueError("Count contains no observed passenger counts.")
    source = pd.DataFrame({"Datetime": dates, "Count": counts.astype(float)})
    source = source.set_index("Datetime").sort_index()
    validate_time_index(source, require_hourly=False)
    timeline = pd.date_range(source.index.min(), source.index.max(), freq="h", name="Datetime")
    missing_hours = timeline.difference(source.index)
    print(f"Loaded observations: {len(source):,}; missing source counts: {source['Count'].isna().sum():,}")
    print(f"Missing hourly timestamps inserted as NaN: {len(missing_hours):,}")
    if len(missing_hours):
        print("First missing timestamps:", ", ".join(missing_hours[:5].astype(str)))
    regular = source.reindex(timeline)
    validate_time_index(regular)
    return regular


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with calendar, lag, and complete historical rolling features."""
    validate_time_index(data)
    result = data[["Count"]].copy()
    dates = result.index
    for name, values in {
        "hour": dates.hour, "day_of_week": dates.dayofweek,
        "day_of_month": dates.day, "month": dates.month,
        "year": dates.year, "is_weekend": dates.dayofweek >= 5,
        "week_of_year": dates.isocalendar().week.to_numpy(),
    }.items():
        result[name] = np.asarray(values, dtype=np.int16 if name == "year" else np.int8)
    for lag in LAGS:
        # lag_24 is the same hour yesterday; lag_168 is that hour last week.
        result[f"lag_{lag}"] = result["Count"].shift(lag)
    history = result["Count"].shift(1)  # Exclude the current target BEFORE rolling.
    for window in (24, 168):
        rolling = history.rolling(window=window, min_periods=window)
        result[f"rolling_mean_{window}"] = rolling.mean()
        result[f"rolling_std_{window}"] = rolling.std(ddof=1)
        if window == 24:
            result["rolling_min_24"] = rolling.min()
            result["rolling_max_24"] = rolling.max()
    return result[["Count"] + FEATURE_COLUMNS]


def prepare_modeling_dataset(features: pd.DataFrame) -> pd.DataFrame:
    """Report missingness and retain only complete, finite modeling rows."""
    print("\nMissing values by target and feature:\n" + features.isna().sum().to_string())
    missing_features = features[FEATURE_COLUMNS].isna().any(axis=1)
    missing_targets = features["Count"].isna()
    print(f"Rows without sufficient historical features (warm-up or gaps): {missing_features.sum():,}")
    print(f"Rows with missing targets: {missing_targets.sum():,}")
    print("These two counts can overlap.")
    modeling = features.loc[~(missing_features | missing_targets)].copy()
    print(f"Rows remaining after filtering: {len(modeling):,} of {len(features):,}")
    if modeling.empty:
        raise ValueError("No complete modeling rows remain; at least 168 hours of observed history are needed.")
    if not np.isfinite(modeling[["Count"] + FEATURE_COLUMNS].to_numpy()).all():
        raise ValueError("Modeling data contains missing or nonfinite values.")
    return modeling


def chronological_split(
    data: pd.DataFrame, train_fraction: float = 0.70, val_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split rows in time order; the test set receives the rounding remainder.

    Filtered rows can have gaps. Fractions refer to retained rows, not duration.
    No transformation is fitted, and historical context may cross split boundaries.
    """
    validate_time_index(data, require_hourly=False)
    if not (0 < train_fraction < 1 and 0 < val_fraction < 1 and train_fraction + val_fraction < 1):
        raise ValueError("Split fractions must be positive and sum to less than one.")
    train_end = int(len(data) * train_fraction)
    val_end = train_end + int(len(data) * val_fraction)
    splits = (data.iloc[:train_end].copy(), data.iloc[train_end:val_end].copy(), data.iloc[val_end:].copy())
    if any(part.empty for part in splits):
        raise ValueError("Too few complete rows for three nonempty chronological splits.")
    train, val, test = splits
    if not (train.index.max() < val.index.min() and val.index.max() < test.index.min()):
        raise ValueError("Train, validation, and test periods overlap.")
    for name, part in zip(("Training", "Validation", "Testing"), splits):
        print(f"{name}: {len(part):,} rows, {part.index.min()} to {part.index.max()}")
    print("Verified: split periods do not overlap.")
    return splits


def test_feature_behavior() -> None:
    """Check exact historical values, gaps, and current/future target isolation."""
    index = pd.date_range("2020-01-01", periods=400, freq="h", name="Datetime")
    source = pd.DataFrame({"Count": np.arange(400, dtype=float)}, index=index)
    features = add_features(source)
    position = 200
    for lag in LAGS:
        assert features.loc[index[position], f"lag_{lag}"] == source.loc[index[position] - pd.Timedelta(hours=lag), "Count"]
    for window in (24, 168):
        history = source["Count"].iloc[position - window:position]
        assert np.isclose(features.iloc[position][f"rolling_mean_{window}"], history.mean())
        assert np.isclose(features.iloc[position][f"rolling_std_{window}"], history.std(ddof=1))
    assert features.iloc[position]["rolling_min_24"] == 176
    assert features.iloc[position]["rolling_max_24"] == 199
    changed = source.copy()
    changed.iloc[position:, 0] = 100000
    pd.testing.assert_frame_equal(
        features.iloc[:position + 1][FEATURE_COLUMNS],
        add_features(changed).iloc[:position + 1][FEATURE_COLUMNS],
    )
    # Remove a timestamp and restore the hourly grid, exactly as loading does.
    gapped = source.drop(index[190]).reindex(index)
    gap_features = add_features(gapped)
    assert pd.isna(gap_features.loc[index[191], "lag_1"])
    assert gap_features.loc[index[192], "lag_1"] == 191
    assert pd.isna(gap_features.loc[index[214], "lag_24"])
    assert pd.isna(gap_features.loc[index[200], "rolling_mean_24"])
    assert pd.isna(gap_features.loc[index[358], "rolling_mean_168"])
    assert pd.notna(gap_features.loc[index[359], "rolling_mean_168"])
    print("Synthetic lag, rolling, gap, and leakage checks passed.")


def main() -> int:
    """Validate, engineer, filter, split, and save datasets without model training."""
    try:
        test_feature_behavior()
        data = load_dataset(DATA_DIRECTORY / "train.csv")
        features = add_features(data)
        modeling = prepare_modeling_dataset(features)
        train, val, test = chronological_split(modeling)
        # The full feature table preserves NaNs for inspection; split files are complete.
        for filename, frame in (
            ("passenger_features.csv", features), ("train_features.csv", train),
            ("val_features.csv", val), ("test_features.csv", test),
        ):
            path = DATA_DIRECTORY / filename
            frame.to_csv(path, index=True, index_label="Datetime", date_format="%Y-%m-%d %H:%M:%S")
            print(f"Saved {path} ({len(frame):,} rows)")
        print("One-step-ahead setup: actual demand through t-1 is available at each prediction.")
        print("For multi-step forecasts, construct features using only information available at the forecast origin.")
    except (OSError, ValueError, pd.errors.ParserError) as error:
        print(f"\nFeature preparation could not complete: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
