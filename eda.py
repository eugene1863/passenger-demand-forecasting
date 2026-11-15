"""Explore JetRail hourly passenger demand without changing the source CSV.

Run from the project directory with ``python eda.py``. Install dependencies with
``python -m pip install pandas numpy matplotlib seaborn`` if needed.
Charts describe observed patterns; they do not establish stable seasonality.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save figures reliably, including in terminals without a GUI.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BASE_DIRECTORY = Path(__file__).resolve().parent
DATA_PATH = BASE_DIRECTORY / "data" / "train.csv"
FIGURE_DIRECTORY = BASE_DIRECTORY / "outputs" / "figures"
DATETIME_FORMAT = "%d-%m-%Y %H:%M"
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def load_dataset(path: Path) -> pd.DataFrame:
    """Read and validate required columns, timestamps, and numeric counts.

    Missing counts remain missing. Invalid values raise actionable errors instead
    of being silently dropped or converted into plausible observations.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}. Place the JetRail training CSV at this location.")
    try:
        data = pd.read_csv(path)
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"Dataset is empty: {path}. Replace it with the JetRail training CSV.") from error
    required_columns = {"ID", "Datetime", "Count"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing_columns))}.")
    if data.empty:
        raise ValueError("The CSV contains column headers but no observations.")

    timestamps = pd.to_datetime(data["Datetime"], format=DATETIME_FORMAT, errors="coerce")
    invalid_dates = timestamps.isna()
    if invalid_dates.any():
        examples = data.loc[invalid_dates, "Datetime"].head(5).tolist()
        raise ValueError(
            f"Found {invalid_dates.sum()} missing or invalid Datetime values. "
            f"Expected DD-MM-YYYY HH:MM (for example 25-08-2012 00:00). Examples: {examples}"
        )
    counts = pd.to_numeric(data["Count"], errors="coerce")
    invalid_counts = data["Count"].notna() & counts.isna()
    nonfinite_counts = counts.notna() & ~np.isfinite(counts)
    if invalid_counts.any() or nonfinite_counts.any():
        raise ValueError("Count contains nonnumeric or infinite values. Correct these values in the input CSV.")
    if not counts.notna().any():
        raise ValueError("Count has no observed numeric values to analyze.")
    data = data.copy()
    data["Datetime"] = timestamps
    data["Count"] = counts
    return data.sort_values("Datetime", kind="stable").reset_index(drop=True)


def add_calendar_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create calendar descriptors in memory for grouping and later feature ideas."""
    data = data.copy()
    calendar = data["Datetime"].dt
    data["hour"] = calendar.hour
    data["day_of_week"] = calendar.dayofweek  # Monday = 0; Sunday = 6.
    data["day_of_month"] = calendar.day
    data["month"] = calendar.month
    data["year"] = calendar.year
    data["is_weekend"] = calendar.dayofweek >= 5
    return data


def inspect_dataset(data: pd.DataFrame) -> dict:
    """Print inspection results and identify gaps on the expected hourly grid."""
    source = data[["ID", "Datetime", "Count"]]
    print("\nFirst five rows (chronological):\n", source.head().to_string(index=False))
    print("\nLast five rows (chronological):\n", source.tail().to_string(index=False))
    print(f"\nDataset dimensions: {source.shape[0]:,} rows, {source.shape[1]} source columns")
    print("\nColumn data types:\n", source.dtypes.to_string())
    print("\nPassenger count statistics:\n", source["Count"].describe().to_string())
    missing_values = source.isna().sum()
    timestamps = pd.DatetimeIndex(data["Datetime"].drop_duplicates())
    expected_hours = pd.date_range(timestamps.min(), timestamps.max(), freq="h")
    missing_hours = expected_hours.difference(timestamps)
    off_grid = timestamps.difference(expected_hours)
    intervals = timestamps.to_series().diff().dropna()
    irregular_intervals = int(intervals.ne(pd.Timedelta(hours=1)).sum())
    duplicate_count = int(data["Datetime"].duplicated().sum())
    negative_count = int(data["Count"].lt(0).sum())
    print("\nMissing values:\n", missing_values.to_string())
    print(f"Duplicate timestamp rows beyond the first occurrence: {duplicate_count:,}")
    print(f"Negative passenger counts: {negative_count:,}")
    print(f"Non-hourly intervals between unique timestamps: {irregular_intervals:,}")
    print(f"Missing hourly timestamps: {len(missing_hours):,}")
    print(f"Timestamps off the hourly grid anchored at the first observation: {len(off_grid):,}")
    if len(missing_hours):
        print("First missing hours:", ", ".join(missing_hours[:10].astype(str)))
    if duplicate_count:
        print("CAUTION: totals and distributions include all recorded rows, including duplicates.")
    if negative_count:
        print("CAUTION: negative counts are retained for investigation; they may distort summaries.")
    if len(missing_hours) or data["Count"].isna().any():
        print("CAUTION: daily/monthly totals may be incomplete. No passenger counts are imputed.")
    return {
        "missing_values": missing_values,
        "missing_hours": missing_hours,
        "duplicate_count": duplicate_count,
        "negative_count": negative_count,
    }


def save_figure(figure: plt.Figure, directory: Path, filename: str) -> None:
    """Save a consistently formatted PNG and release its plotting resources."""
    directory.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(directory / filename, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_time_series(data: pd.DataFrame, directory: Path) -> None:
    """Plot hourly observations and daily/monthly sums, leaving empty bins missing."""
    series = data.set_index("Datetime")["Count"]
    daily_totals = series.resample("D").sum(min_count=1)
    monthly_totals = series.resample("MS").sum(min_count=1)
    # A missing hour stays NaN so the hourly line breaks instead of bridging a gap.
    unique_counts = series.groupby(level=0).mean()
    hourly_grid = pd.date_range(series.index.min(), series.index.max(), freq="h")
    hourly_line = unique_counts.reindex(unique_counts.index.union(hourly_grid)).sort_index()
    for values, title, ylabel, filename in [
        (hourly_line, "Hourly passenger demand (duplicate timestamps averaged)", "Passengers per observation", "01_hourly_demand.png"),
        (daily_totals, "Daily passenger demand — sum of recorded hourly counts", "Passengers per day (sum)", "02_daily_total_demand.png"),
        (monthly_totals, "Monthly passenger demand — sum of recorded hourly counts", "Passengers per month (sum)", "03_monthly_total_demand.png"),
    ]:
        figure, axis = plt.subplots(figsize=(13, 5))
        axis.plot(values.index, values.values, linewidth=1)
        axis.set(title=title, xlabel="Date", ylabel=ylabel)
        save_figure(figure, directory, filename)
    print("Daily/monthly totals use available counts; boundary periods may be partial.")


def plot_calendar_patterns(data: pd.DataFrame, directory: Path) -> None:
    """Compare pooled hourly averages by calendar groups, plus hourly spread."""
    for column, order, labels, title, filename in [
        ("hour", range(24), list(range(24)), "Hour of day", "05_average_by_hour.png"),
        ("day_of_week", range(7), DAY_NAMES, "Day of week", "06_average_by_weekday.png"),
        ("month", range(1, 13), MONTH_NAMES, "Month", "07_average_by_month.png"),
        ("is_weekend", [False, True], ["Weekday", "Weekend"], "Weekday versus weekend", "08_weekday_weekend.png"),
    ]:
        averages = data.groupby(column)["Count"].mean().reindex(order)
        figure, axis = plt.subplots(figsize=(11, 5))
        axis.bar(np.arange(len(averages)), averages.values)
        axis.set_xticks(np.arange(len(averages)), labels, rotation=30 if column == "day_of_week" else 0)
        axis.set(title=f"Average hourly passenger demand — {title.lower()}", xlabel=title,
                 ylabel="Mean passengers per hourly observation")
        save_figure(figure, directory, filename)
    figure, axis = plt.subplots(figsize=(13, 5))
    sns.boxplot(data=data, x="hour", y="Count", order=list(range(24)), ax=axis, color="#80b1d3")
    axis.set(title="Hourly passenger count distribution by hour of day", xlabel="Hour of day", ylabel="Passengers")
    save_figure(figure, directory, "09_boxplot_by_hour.png")


def plot_rolling_patterns(data: pd.DataFrame, directory: Path) -> None:
    """Use trailing time windows, including only current and past observations.

    Duplicate timestamps are averaged for this chart only. Minimum observation
    counts suppress incomplete windows on regular hourly data. No gap is filled
    with an invented count. Time windows are open on the left, closed on the right.
    """
    hourly_counts = data.groupby("Datetime")["Count"].mean().sort_index()
    grid = pd.date_range(hourly_counts.index.min(), hourly_counts.index.max(), freq="h")
    hourly_counts = hourly_counts.reindex(hourly_counts.index.union(grid)).sort_index()
    rolling_day = hourly_counts.rolling("24h", min_periods=24, closed="right").mean()
    rolling_week = hourly_counts.rolling("168h", min_periods=168, closed="right").mean()
    figure, axis = plt.subplots(figsize=(13, 6))
    axis.plot(hourly_counts.index, hourly_counts, alpha=0.3, linewidth=0.7, label="Hourly observations")
    axis.plot(rolling_day.index, rolling_day, linewidth=1.3, label="Trailing 24-hour mean")
    axis.plot(rolling_week.index, rolling_week, linewidth=2, label="Trailing 7-day mean")
    axis.set(title="Observed demand and trailing rolling averages", xlabel="Date", ylabel="Passengers per hour (mean)")
    axis.legend()
    save_figure(figure, directory, "10_rolling_averages.png")
    print("Rolling means require 24/168 nonmissing timestamps per window; short or incomplete windows stay blank.")
    print("Interpret those windows as hourly averages only after resolving any off-grid timestamps.")


def analyze_outliers(data: pd.DataFrame, directory: Path) -> pd.Series:
    """Flag counts beyond global 1.5-IQR fences without deleting observations."""
    counts = data["Count"]
    first_quartile, third_quartile = counts.quantile([0.25, 0.75])
    interquartile_range = third_quartile - first_quartile
    lower_fence = first_quartile - 1.5 * interquartile_range
    upper_fence = third_quartile + 1.5 * interquartile_range
    flagged = counts.lt(lower_fence) | counts.gt(upper_fence)
    valid_count = int(counts.notna().sum())
    print(f"\nIQR fences: {lower_fence:,.2f} to {upper_fence:,.2f}")
    print(f"Potential outliers: {flagged.sum():,} / {valid_count:,} observed counts ({100 * flagged.sum() / valid_count:.2f}%)")
    print("Global IQR flags can reflect growth, calendar patterns, or real demand spikes; no rows are removed.")
    figure, axis = plt.subplots(figsize=(11, 5))
    sns.histplot(counts.dropna(), bins=40, ax=axis, color="#377eb8")
    axis.set(title="Distribution of hourly passenger counts", xlabel="Passengers per hourly observation", ylabel="Number of observations")
    save_figure(figure, directory, "04_hourly_count_histogram.png")
    figure, axes = plt.subplots(2, 1, figsize=(11, 7))
    sns.histplot(counts.dropna(), bins=40, ax=axes[0], color="#377eb8")
    axes[0].axvline(lower_fence, color="#e41a1c", linestyle="--", label="IQR fences")
    axes[0].axvline(upper_fence, color="#e41a1c", linestyle="--")
    axes[0].legend()
    axes[0].set(title="Potentially unusual hourly counts — global IQR rule", xlabel="Passengers", ylabel="Observations")
    sns.boxplot(x=counts.dropna(), ax=axes[1], color="#80b1d3")
    axes[1].set(xlabel="Passengers (boxplot points are potential outliers)")
    save_figure(figure, directory, "11_iqr_outlier_distribution.png")
    return flagged


def print_summary(data: pd.DataFrame, quality: dict, outliers: pd.Series) -> None:
    """Print a concise final report and cautious forecasting interpretation."""
    counts = data["Count"]
    print("\n=== EDA summary ===")
    print(f"Observations: {len(data):,}")
    print(f"Date range: {data['Datetime'].min()} to {data['Datetime'].max()}")
    print(f"Missing hourly timestamps: {len(quality['missing_hours']):,}")
    print("Missing values: " + ", ".join(f"{column}={value:,}" for column, value in quality["missing_values"].items()))
    print(f"Duplicate timestamp rows (beyond first): {quality['duplicate_count']:,}")
    print(f"Negative counts: {quality['negative_count']:,}")
    print(f"Passenger counts: min={counts.min():,.2f}, max={counts.max():,.2f}, mean={counts.mean():,.2f}")
    print(f"Potential IQR outliers: {outliers.sum():,}")
    print(f"Figures saved to: {FIGURE_DIRECTORY}")
    print("\nPattern exploration:")
    print("- Compare hour-of-day means for possible daily patterns and weekday means for weekly patterns.")
    print("- Compare monthly totals and the 7-day rolling mean for longer-term changes in demand.")
    print("- Trends change the demand baseline; repeating calendar patterns may inform seasonal features.")
    print("- Grouped averages alone do not establish seasonality: growth, unequal coverage, and partial periods can confound them.")
    print("- Validate candidate patterns across time and with chronological backtesting before forecasting.")


def main() -> int:
    """Run validation, exploratory plots, and summary; return a terminal exit code."""
    try:
        data = load_dataset(DATA_PATH)
        quality = inspect_dataset(data)
        data = add_calendar_features(data)
        FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid", palette="deep")
        plot_time_series(data, FIGURE_DIRECTORY)
        plot_calendar_patterns(data, FIGURE_DIRECTORY)
        plot_rolling_patterns(data, FIGURE_DIRECTORY)
        outliers = analyze_outliers(data, FIGURE_DIRECTORY)
        print_summary(data, quality, outliers)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        print(f"\nEDA could not complete: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
