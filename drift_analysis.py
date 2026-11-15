"""Retrospective demand/feature shift diagnostics; run python drift_analysis.py.

Uses the trusted local model and existing predictions without fitting or predicting.
Shared artifact/schema readers come from evaluate.py; its main() is never invoked.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from evaluate import load_data, load_model

BASE = Path(__file__).resolve().parent
FEATURES = ["lag_1", "lag_24", "lag_168", "rolling_mean_24", "rolling_mean_168", "rolling_std_24"]
COLORS = {"Training": "#377eb8", "Validation": "#ff7f00", "Test": "#4daf4a"}


def statistics(values: pd.Series) -> dict:
    """Return descriptive statistics; standard deviation uses ddof=1."""
    return {"Observations": len(values), "Mean": values.mean(), "Median": values.median(),
            "Std": values.std(ddof=1), "Minimum": values.min(), "Maximum": values.max(),
            "P25": values.quantile(.25), "P75": values.quantile(.75), "P95": values.quantile(.95)}


def load_inputs() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Validate model schema, ordered splits, and exact prediction/test alignment."""
    model, columns, metadata = load_model()
    missing = set(FEATURES) - set(columns)
    if missing:
        raise ValueError(f"Expected diagnostic features absent from model schema: {sorted(missing)}")
    splits = {label: load_data(BASE / "data" / f"{name}_features.csv", columns)
              for label, name in (("Training", "train"), ("Validation", "val"), ("Test", "test"))}
    train, val, test = splits.values()
    if not train["Datetime"].iloc[-1] < val["Datetime"].iloc[0] or not val["Datetime"].iloc[-1] < test["Datetime"].iloc[0]:
        raise ValueError("Train/validation/test dates overlap or are out of order.")
    for key, actual in (("training_start", train["Datetime"].iloc[0]), ("training_end", train["Datetime"].iloc[-1]),
                        ("validation_start", val["Datetime"].iloc[0]), ("validation_end", val["Datetime"].iloc[-1])):
        if key in metadata and pd.Timestamp(metadata[key]) != actual:
            raise ValueError(f"Saved model metadata disagrees with {key}.")
    predictions = pd.read_csv(BASE / "outputs" / "test_predictions.csv")
    required = {"Datetime", "Actual Count", "Predicted Count", "Residual", "Absolute Error"}
    if predictions.empty or required - set(predictions.columns):
        raise ValueError("Missing prediction columns or empty predictions CSV.")
    predictions["Datetime"] = pd.to_datetime(predictions["Datetime"], format="%Y-%m-%d %H:%M:%S", errors="raise")
    numeric = sorted(required - {"Datetime"})
    predictions[numeric] = predictions[numeric].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(predictions[numeric].to_numpy()).all():
        raise ValueError("Predictions contain nonfinite numbers.")
    if not predictions["Datetime"].equals(test["Datetime"]) or not np.allclose(predictions["Actual Count"], test["Count"], rtol=0, atol=1e-9):
        raise ValueError("Predictions do not align exactly with test timestamps/targets.")
    if predictions[["Actual Count", "Predicted Count"]].lt(0).any().any():
        raise ValueError("Predictions or actual counts are negative.")
    residual = predictions["Actual Count"] - predictions["Predicted Count"]
    if not np.allclose(predictions["Residual"], residual) or not np.allclose(predictions["Absolute Error"], residual.abs()):
        raise ValueError("Saved residuals or absolute errors are inconsistent.")
    predictions["Bias"] = -residual  # MBE = predicted - actual.
    importance_values = np.asarray(model.feature_importances_, dtype=float)
    if importance_values.shape != (len(columns),) or not np.isfinite(importance_values).all():
        raise ValueError("Feature importance is incompatible with the saved schema.")
    importance = pd.DataFrame({"Feature": columns, "Importance": importance_values}).sort_values("Importance", ascending=False, kind="stable")
    return splits, predictions, importance


def extrapolation_report(splits: dict) -> pd.DataFrame:
    """Measure training-range exceedances without labeling them concept drift."""
    rows = []
    for feature in ["Count", *FEATURES]:
        train, val, test = (frame[feature] for frame in splits.values())
        lower, upper = train.min(), train.max()
        rows.append({"Feature": feature, "Training Minimum": lower, "Training Maximum": upper,
                     "Validation Maximum": val.max(), "Test Maximum": test.max(),
                     "Test Above Training Maximum (%)": 100 * test.gt(upper).mean(),
                     "Test Below Training Minimum (%)": 100 * test.lt(lower).mean(),
                     "Test Outside Training Range (%)": 100 * (test.lt(lower) | test.gt(upper)).mean()})
    return pd.DataFrame(rows)


def aggregate_errors(predictions: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Aggregate available observations; weekly bins end Sunday, monthly bins start on day one."""
    return predictions.set_index("Datetime").resample(frequency).agg(
        Observations=("Actual Count", "size"), Mean_Actual=("Actual Count", "mean"),
        Mean_Predicted=("Predicted Count", "mean"), MAE=("Absolute Error", "mean"),
        MBE=("Bias", "mean")).rename_axis("Period").reset_index()


def demand_groups(predictions: pd.DataFrame) -> pd.DataFrame:
    """Use actual test targets only to describe errors; preserve ties in quantile bins."""
    frame = predictions.copy()
    bins, edges = pd.qcut(frame["Actual Count"], 4, duplicates="drop", retbins=True)
    if len(edges) < 2:
        frame["Demand Group"] = "Constant actual demand"
    else:
        frame["Demand Group"] = bins.cat.rename_categories([f"Q{i+1}: {b}" for i, b in enumerate(bins.cat.categories)])
    print("Descriptive test demand quantile edges:", edges)
    return frame.groupby("Demand Group", observed=True).agg(
        Observations=("Actual Count", "size"), Mean_Actual=("Actual Count", "mean"),
        Mean_Predicted=("Predicted Count", "mean"), MAE=("Absolute Error", "mean"),
        MBE=("Bias", "mean")).reset_index()


def save_figures(splits: dict, predictions: pd.DataFrame, importance: pd.DataFrame,
                 monthly_demand: pd.DataFrame, monthly: pd.DataFrame, weekly: pd.DataFrame,
                 groups: pd.DataFrame, extrapolation: pd.DataFrame) -> None:
    """Save eight diagnostics with shared scales and data-derived split boundaries."""
    directory = BASE / "outputs" / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(directory / f"drift_{name}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for ax, column in zip(axes.flat, ["Mean", "Median", "Maximum", "Std"]):
        ax.plot(monthly_demand["Datetime"], monthly_demand[column], marker="o", markersize=3, label=column)
        for label in ("Validation", "Test"):
            ax.axvline(splits[label]["Datetime"].iloc[0], color=COLORS[label], linestyle="--", label=f"{label} starts")
        ax.set(title=f"Monthly {column.lower()} hourly demand", xlabel="Month", ylabel="Passengers")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)
    fig.suptitle("Observed modeling rows: monthly demand (boundary months may be partial)")
    save(fig, "monthly_demand")
    fig, ax = plt.subplots(figsize=(11, 5))
    edges = np.histogram_bin_edges(np.concatenate([d["Count"].to_numpy() for d in splits.values()]), bins=45)
    for label, frame in splits.items():
        ax.hist(frame["Count"], bins=edges, density=True, histtype="step", linewidth=1.8, label=label, color=COLORS[label])
    ax.set(title="Passenger demand distributions across chronological splits", xlabel="Hourly passengers", ylabel="Probability density")
    ax.legend()
    save(fig, "target_distribution")
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    for ax, feature in zip(axes.flat, FEATURES):
        edges = np.histogram_bin_edges(np.concatenate([d[feature].to_numpy() for d in splits.values()]), bins=40)
        for label, frame in splits.items():
            ax.hist(frame[feature], bins=edges, density=True, histtype="step", linewidth=1.5, label=label, color=COLORS[label])
        ax.set(title=feature, xlabel="Passengers (historical feature)", ylabel="Probability density")
        ax.legend(fontsize=8)
    save(fig, "lag_distributions")
    fig, ax = plt.subplots(figsize=(10, 8))
    ordered = importance.iloc[::-1]
    ax.barh(ordered["Feature"], ordered["Importance"])
    ax.set(title="Saved Random Forest: impurity-based importance", xlabel="Relative importance", ylabel="Feature")
    save(fig, "feature_importance")
    for table, name in ((monthly, "monthly"), (weekly, "weekly")):
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        for column, label in (("MAE", "MAE"), ("Mean_Actual", "Mean actual demand")):
            axes[0].plot(table["Period"], table[column], marker="o", label=label)
        axes[0].set(title=f"{name.capitalize()} test errors and demand (partial boundary bins included)", ylabel="Passengers")
        axes[0].set_ylim(bottom=0)
        axes[0].legend()
        axes[1].plot(table["Period"], table["MBE"], marker="o", label="MBE: predicted - actual")
        axes[1].axhline(0, color="black", linestyle="--")
        axes[1].set(xlabel="Period", ylabel="Mean bias (passengers)")
        axes[1].legend()
        save(fig, f"{name}_mae")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].scatter(weekly["Mean_Actual"], weekly["MAE"], label="Weekly test bins")
    axes[0].set(title="Weekly demand versus absolute error", xlabel="Mean actual passengers", ylabel="MAE (passengers)")
    axes[0].legend()
    x = np.arange(len(groups))
    axes[1].bar(x - .2, groups["MAE"], width=.4, label="MAE")
    axes[1].bar(x + .2, groups["MBE"], width=.4, label="MBE: predicted - actual")
    axes[1].axhline(0, color="black", linewidth=.8)
    axes[1].set_xticks(x, groups["Demand Group"].astype(str), rotation=25, ha="right")
    axes[1].set(title="Retrospective actual-demand groups", xlabel="Test actual-demand quartile", ylabel="Passengers")
    axes[1].legend()
    save(fig, "demand_vs_error")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].scatter(predictions["Actual Count"], predictions["Predicted Count"], s=12, alpha=.3, label="Saved test predictions")
    maximum = max(predictions["Actual Count"].max(), predictions["Predicted Count"].max())
    axes[0].plot([0, maximum], [0, maximum], "k--", label="Perfect prediction")
    ceiling = splits["Training"]["Count"].max()
    axes[0].axhline(ceiling, linestyle=":", color="red", label="Maximum training target")
    axes[0].axvline(ceiling, linestyle=":", color="red")
    axes[0].set(xlabel="Actual passengers", ylabel="Predicted passengers", title="Test predictions versus training target range", xlim=(0, maximum * 1.05), ylim=(0, maximum * 1.05))
    axes[0].legend(fontsize=8)
    axes[1].barh(extrapolation["Feature"], extrapolation["Test Above Training Maximum (%)"])
    axes[1].set(xlabel="Test observations above training maximum (%)", title="Upper training-range exceedances", xlim=(0, 100))
    save(fig, "extrapolation")


def print_report(target: pd.DataFrame, extrapolation: pd.DataFrame, importance: pd.DataFrame,
                 monthly: pd.DataFrame, weekly: pd.DataFrame, monthly_demand: pd.DataFrame) -> None:
    """Report descriptive evidence and explicitly separate shift from concept drift."""
    print("\nTarget distributions:\n" + target.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nTraining-range checks:\n" + extrapolation.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nFive most important features:\n" + importance.head(5).to_string(index=False))
    print("\nMonthly test errors; MBE = predicted - actual:\n" + monthly.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    correlation = weekly["Mean_Actual"].corr(weekly["MAE"]) if len(weekly) > 1 else np.nan
    print(f"\nWeekly mean demand / MAE Pearson correlation: {correlation:.4f} (descriptive, not causal).")
    means = target["Mean"].to_numpy()
    if np.all(np.diff(means) > 0):
        print("Mean demand increases from training to validation to test; evidence supports an upward target-distribution shift.")
    elif np.all(np.diff(means) < 0):
        print("Mean demand decreases across splits; evidence supports a downward target-distribution shift.")
    else:
        print("Split means do not change monotonically; inspect the distribution summaries for the direction and scale of changes.")
    print(f"First/last available monthly mean: {monthly_demand['Mean'].iloc[0]:.3f} / {monthly_demand['Mean'].iloc[-1]:.3f} passengers; intermediate months can fluctuate.")
    print("Monthly statistics cover retained modeling observations, excluding warm-up/missing-feature rows; boundary months/weeks may be partial.")
    print("Random Forest regression averages training-target values in leaves and across trees, so it generally cannot extrapolate beyond the training target range.")
    print("Actual predictions may occupy a much narrower range than the maximum training target.")
    print("Out-of-range inputs and changing target marginals do not prove concept drift (a change in P(target | features)).")
    print("Additional conditional/time-controlled analysis is needed; growth and calendar composition can also explain distribution changes.")
    print("Feature importances describe impurity reduction, not causality; correlated lag features can share or distort importance.")
    print("No significance tests or arbitrary drift thresholds are used; serial dependence would complicate ordinary two-sample p-values.")
    print("Negative MBE means underprediction; positive MBE means overprediction. Demand-group targets are retrospective diagnostics only.")
    print("No fitting, tuning, model selection, or changes to prior predictions occurred. The test period is now diagnostic evidence.")
    print("Any redesigned model requires earlier chronological validation or rolling backtesting; a genuinely new future holdout is needed for an independent final assessment.")
    print("Existing results represent rolling one-step-ahead predictions, not a fixed-origin future-week forecast.")


def main() -> int:
    """Read existing artifacts and write only drift reports and figures."""
    try:
        print("Loading trusted local model, chronological splits, and saved test predictions...")
        splits, predictions, importance = load_inputs()
        target = pd.DataFrame([{"Split": label, **statistics(frame["Count"])} for label, frame in splits.items()])
        feature_stats = pd.DataFrame([{"Feature": feature, "Split": label, **statistics(frame[feature])}
                                     for feature in FEATURES for label, frame in splits.items()])
        extrapolation = extrapolation_report(splits)
        full = pd.concat(list(splits.values()), ignore_index=True)
        monthly_demand = full.set_index("Datetime")["Count"].resample("MS").agg(
            Observations="size", Mean="mean", Median="median", Maximum="max", Std="std").reset_index()
        monthly = aggregate_errors(predictions, "MS")
        weekly = aggregate_errors(predictions, "W-SUN")
        groups = demand_groups(predictions)
        output = BASE / "outputs"
        output.mkdir(parents=True, exist_ok=True)
        for name, table in (("target_statistics", target), ("feature_statistics", feature_stats),
                            ("extrapolation_report", extrapolation), ("feature_importance", importance),
                            ("monthly_errors", monthly), ("weekly_errors", weekly),
                            ("demand_group_errors", groups), ("monthly_demand", monthly_demand)):
            table.to_csv(output / f"drift_{name}.csv", index=False)
        save_figures(splits, predictions, importance, monthly_demand, monthly, weekly, groups, extrapolation)
        print_report(target, extrapolation, importance, monthly, weekly, monthly_demand)
        print(f"Saved drift reports and eight figures under {output}.")
    except (OSError, ValueError, TypeError, KeyError, AttributeError, ImportError) as error:
        print(f"Drift analysis could not complete: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
