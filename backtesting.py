"""Historical rolling one-step backtests; run python backtesting.py --help.

Only data before the immutable original test cutoff are used. Earlier observed
validation counts may supply features at later hours, but never training labels
for that block. Forests are refitted once per block, not once per hour.
"""
import argparse
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.ensemble import RandomForestRegressor

from features import FEATURE_COLUMNS, add_features, prepare_modeling_dataset, validate_time_index

BASE = Path(__file__).resolve().parent
CUTOFF = pd.Timestamp("2014-06-04 18:00:00")
STRATEGIES = ("Previous hour", "RF expanding", "RF sliding", "RF delta expanding")
RF_PARAMS = dict(n_estimators=200, max_depth=15, min_samples_leaf=2,
                 random_state=42, n_jobs=4)
ASSUMPTION = (
    "Rolling one-step-ahead: actual counts through t-1 are available at t, including "
    "earlier actuals within validation blocks. This does not measure fixed-origin "
    "24-hour forecasting. Models are refitted only at each block start."
)


def load_history() -> pd.DataFrame:
    """Filter at cutoff before numeric checks; audit saved features from past counts."""
    frame = pd.read_csv(BASE / "data" / "passenger_features.csv")
    required = ["Datetime", "Count", *FEATURE_COLUMNS]
    if set(required) - set(frame.columns):
        raise ValueError("Passenger features CSV has an incomplete schema.")
    frame["Datetime"] = pd.to_datetime(frame["Datetime"], format="%Y-%m-%d %H:%M:%S", errors="raise")
    # No post-cutoff target or feature is inspected or used in development.
    frame = frame.loc[frame["Datetime"] < CUTOFF, required].set_index("Datetime")
    validate_time_index(frame, require_hourly=False)
    frame = frame.apply(pd.to_numeric, errors="raise")
    values = frame.to_numpy(dtype=float)
    if np.isinf(values).any() or frame["Count"].lt(0).any():
        raise ValueError("Historical data contain infinity or negative targets.")
    grid = pd.date_range(frame.index.min(), frame.index.max(), freq="h", name="Datetime")
    recomputed = add_features(frame[["Count"]].reindex(grid))
    expected = recomputed.reindex(frame.index)[FEATURE_COLUMNS].to_numpy(dtype=float)
    if not np.allclose(frame[FEATURE_COLUMNS].to_numpy(dtype=float), expected,
                       rtol=1e-7, atol=1e-7, equal_nan=True):
        raise ValueError("Saved features disagree with historical-only feature definitions.")
    print("Verified all saved calendar/lag/rolling features against pre-test counts.")
    return prepare_modeling_dataset(recomputed)


def make_folds(history: pd.DataFrame, folds: int, block_days: int,
               window_days: int, min_history_days: int) -> list[dict]:
    """Use consecutive duration-based blocks ending at the immutable cutoff."""
    if min(folds, block_days, window_days, min_history_days) <= 0:
        raise ValueError("Fold counts and day durations must be positive.")
    first = CUTOFF - pd.Timedelta(days=folds * block_days)
    if first - history.index.min() < pd.Timedelta(days=max(window_days, min_history_days)):
        raise ValueError("Insufficient retained history before first fold; reduce fold count/durations.")
    result = []
    for number in range(1, folds + 1):
        start = first + pd.Timedelta(days=(number - 1) * block_days)
        end = start + pd.Timedelta(days=block_days)
        validation = history.loc[(history.index >= start) & (history.index < end)]
        expanding = history.loc[history.index < start]
        sliding = expanding.loc[expanding.index >= start - pd.Timedelta(days=window_days)]
        if min(len(validation), len(expanding), len(sliding)) < 2:
            raise ValueError(f"Fold {number} has too few complete observations.")
        if not expanding.index.max() < validation.index.min() < CUTOFF:
            raise ValueError("Fold chronology check failed.")
        result.append(dict(fold=number, start=start, end=end, validation=validation,
                           expanding=expanding, sliding=sliding))
    return result


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """MBE = predicted minus actual; WAPE is percent and undefined at zero demand."""
    error = predicted - actual
    total = actual.sum()
    return dict(MAE=float(np.abs(error).mean()), RMSE=float(np.sqrt(np.mean(error ** 2))),
                MBE=float(error.mean()), WAPE_pct=float(100 * np.abs(error).sum() / total)
                if total > 0 else np.nan)


def run_backtests(folds: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit historical models and evaluate every strategy on identical timestamps."""
    results, predictions = [], []
    for fold in folds:
        validation = fold["validation"]
        actual = validation["Count"].to_numpy(dtype=float)
        for strategy in STRATEGIES:
            train = fold["sliding" if strategy == "RF sliding" else "expanding"]
            if strategy == "Previous hour":
                predicted = validation["lag_1"].to_numpy(dtype=float)
            else:
                target = train["Count"]
                if strategy == "RF delta expanding":
                    target = target - train["lag_1"]
                model = RandomForestRegressor(**RF_PARAMS)
                model.fit(train[FEATURE_COLUMNS], target)
                predicted = model.predict(validation[FEATURE_COLUMNS])
                if strategy == "RF delta expanding":
                    predicted = predicted + validation["lag_1"].to_numpy()
                predicted = np.clip(predicted, 0, None)
            if predicted.shape != actual.shape or not np.isfinite(predicted).all():
                raise ValueError("Invalid model predictions.")
            row = dict(Fold=fold["fold"], Strategy=strategy,
                       Training_Start=train.index.min(), Training_End=train.index.max(),
                       Training_Observations=len(train) if strategy != "Previous hour" else 0,
                       Validation_Block_Start=fold["start"], Validation_Block_End_Exclusive=fold["end"],
                       Validation_Start=validation.index.min(), Validation_End=validation.index.max(),
                       Observations=len(validation), Mean_Actual=float(actual.mean()),
                       **metrics(actual, predicted))
            results.append(row)
            predictions.append(pd.DataFrame(dict(Fold=fold["fold"], Strategy=strategy,
                Datetime=validation.index, Actual_Count=actual, Predicted_Count=predicted,
                Residual=actual - predicted, Absolute_Error=np.abs(actual - predicted))))
            print(f"Fold {fold['fold']} | {strategy}: train {row['Training_Start']} to "
                  f"{row['Training_End']} ({row['Training_Observations']:,} fit rows); "
                  f"validate {row['Validation_Start']} to {row['Validation_End']} "
                  f"({len(validation):,}); MAE={row['MAE']:.3f}, MBE={row['MBE']:.3f}", flush=True)
    return pd.DataFrame(results), pd.concat(predictions, ignore_index=True)


def summarize(results: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Rank by pooled historical MAE, then RMSE and strategy name to break ties."""
    baseline = results.loc[results.Strategy.eq("Previous hour")].set_index("Fold").MAE
    rows = []
    for strategy, frame in predictions.groupby("Strategy", sort=False):
        folds = results.loc[results.Strategy.eq(strategy)].sort_values("Fold")
        correlation = (folds.Mean_Actual.corr(folds.MAE)
                       if folds.Mean_Actual.nunique() > 1 and folds.MAE.nunique() > 1 else np.nan)
        rows.append(dict(Strategy=strategy, Observations=len(frame),
            **metrics(frame.Actual_Count.to_numpy(), frame.Predicted_Count.to_numpy()),
            Fold_Mean_MAE=folds.MAE.mean(), Fold_Std_MAE=folds.MAE.std(ddof=1),
            Worst_Fold_MAE=folds.MAE.max(), First_Fold_MAE=folds.MAE.iloc[0],
            Last_Fold_MAE=folds.MAE.iloc[-1], Demand_MAE_Correlation=correlation,
            Folds_Beating_Baseline=int((folds.set_index("Fold").MAE < baseline).sum()),
            Folds=len(folds)))
    summary = pd.DataFrame(rows).sort_values(["MAE", "RMSE", "Strategy"], kind="stable").reset_index(drop=True)
    baseline_mae = summary.loc[summary.Strategy.eq("Previous hour"), "MAE"].iloc[0]
    summary["MAE_Improvement_vs_Baseline_pct"] = (
        100 * (baseline_mae - summary.MAE) / baseline_mae if baseline_mae > 0 else np.nan)
    summary["Selected"] = summary.index == 0
    return summary


def save_figures(results: pd.DataFrame, predictions: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Save five diagnostics; representative fold chosen by chronological middle."""
    directory = BASE / "outputs" / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(directory / f"backtesting_{name}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(summary.Strategy, summary.MAE)
    ax.invert_yaxis()
    ax.set(title="Pooled historical backtest MAE", xlabel="MAE (passengers)")
    save(fig, "mae_comparison")
    for metric, name, label in (("MAE", "fold_mae", "MAE (passengers)"),
                                ("MBE", "fold_bias", "MBE: predicted - actual")):
        fig, ax = plt.subplots(figsize=(11, 5))
        for strategy, frame in results.groupby("Strategy", sort=False):
            ax.plot(frame.Fold, frame[metric], marker="o", label=strategy)
        if metric == "MBE":
            ax.axhline(0, color="black", linestyle="--")
        ax.set(title=f"Historical {metric} across chronological folds", xlabel="Fold (earlier to later)", ylabel=label)
        ax.legend()
        save(fig, name)
    fold = sorted(predictions.Fold.unique())[len(predictions.Fold.unique()) // 2]
    frame = predictions.loc[predictions.Fold.eq(fold)]
    fig, ax = plt.subplots(figsize=(14, 6))
    actual = frame.loc[frame.Strategy.eq("Previous hour")]
    ax.plot(actual.Datetime, actual.Actual_Count, color="black", label="Actual", linewidth=1.2)
    for strategy, part in frame.groupby("Strategy", sort=False):
        ax.plot(part.Datetime, part.Predicted_Count, label=strategy, alpha=.7, linewidth=.7)
    ax.set(title=f"Chronological middle fold {fold}: rolling one-step predictions", ylabel="Passengers", xlabel="Timestamp")
    ax.legend()
    save(fig, "actual_vs_predicted")
    fig, ax = plt.subplots(figsize=(13, max(5, len(results) * .28)))
    labels = []
    for y, row in enumerate(results.itertuples()):
        labels.append(f"F{row.Fold}: {row.Strategy}")
        if row.Training_Observations:
            left = mdates.date2num(row.Training_Start)
            right = mdates.date2num(row.Training_End + pd.Timedelta(hours=1))
            ax.barh(y, right - left, left=left, color="#377eb8", label="Training" if y == 1 else None)
        left = mdates.date2num(row.Validation_Block_Start)
        ax.barh(y, mdates.date2num(row.Validation_Block_End_Exclusive) - left, left=left,
                color="#ff7f00", label="Validation" if y == 0 else None)
    ax.set_yticks(range(len(labels)), labels)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set(title="Training and validation windows (baseline has no fitting)", xlabel="Date")
    ax.legend()
    save(fig, "training_timeline")


def print_report(results: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Report observed comparisons without asserting significance or concept drift."""
    print(f"\nBacktesting folds: {results.Fold.nunique()}")
    print("\nFold results:\n" + results.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nPooled metrics and fold stability:\n" + summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Best historical strategy: {summary.iloc[0].Strategy}; selected by pooled MAE, then RMSE.")
    indexed = summary.set_index("Strategy")
    for strategy in STRATEGIES[1:]:
        row = indexed.loc[strategy]
        verdict = "beats" if row.MAE < indexed.loc["Previous hour", "MAE"] else "does not beat"
        print(f"{strategy} {verdict} previous-hour pooled MAE; wins {int(row.Folds_Beating_Baseline)}/{int(row.Folds)} folds.")
    expanding = indexed.loc["RF expanding"]
    for strategy, description in (("RF sliding", "Shorter sliding history"),
                                  ("RF delta expanding", "Predicting hourly changes")):
        row = indexed.loc[strategy]
        direction = "improves" if row.MAE < expanding.MAE else "does not improve"
        print(f"{description} {direction} pooled MAE versus expanding absolute-demand RF "
              f"({row.MAE:.3f} vs {expanding.MAE:.3f}); MBE {row.MBE:.3f} vs {expanding.MBE:.3f}; "
              f"absolute pooled bias {'decreases' if abs(row.MBE) < abs(expanding.MBE) else 'does not decrease'}.")
    print("Demand/MAE correlations and first/last/worst fold MAEs describe stability and demand association only; "
          "calendar composition, training size and serial dependence confound causal interpretation.")
    print("Negative MBE means underprediction. WAPE is total absolute error / total actual (%); undefined for zero demand.")
    print("Pooled RMSE is calculated from all squared errors, not averaged fold RMSEs. Baseline uses exactly matching rows.")
    print(ASSUMPTION)
    print("Only one configurable sliding duration is tested per run; this does not identify an optimal window. "
          "Delta RF uses expanding history to isolate the target change. No preprocessing is fitted.")
    print("Original test data at/after 2014-06-04 18:00:00 are excluded from fitting, tuning, ranking and selection.")
    print("This is retrospective development informed by prior test diagnostics; historical backtest selection is optimistic. "
          "A genuinely new independent future holdout is required to verify improvements. "
          "Distribution shift does not establish concept drift; no significance claim is made.")


def main() -> int:
    """Run a reproducible experiment and write only backtesting-prefixed artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--block-days", type=int, default=30)
    parser.add_argument("--window-days", type=int, default=90)
    parser.add_argument("--min-history-days", type=int, default=180)
    args = parser.parse_args()
    try:
        print(ASSUMPTION)
        history = load_history()
        folds = make_folds(history, args.folds, args.block_days, args.window_days, args.min_history_days)
        results, predictions = run_backtests(folds)
        summary = summarize(results, predictions)
        settings = dict(**vars(args), cutoff_exclusive=str(CUTOFF), features=FEATURE_COLUMNS,
            rf_params=RF_PARAMS, delta_training="expanding", refit_frequency="validation block start",
            selection_rule="lowest pooled MAE, then pooled RMSE, then strategy name",
            assumption=ASSUMPTION, python=platform.python_version(), pandas=pd.__version__,
            numpy=np.__version__, sklearn=sklearn.__version__)
        encoded = json.dumps(settings, sort_keys=True)
        results["Experiment_Settings"] = encoded
        selection = summary.copy()
        selection["Experiment_Settings"] = encoded
        output = BASE / "outputs"
        output.mkdir(parents=True, exist_ok=True)
        for name, table in (("fold_results", results), ("predictions", predictions),
                            ("summary", summary), ("strategy_selection", selection)):
            table.to_csv(output / f"backtesting_{name}.csv", index=False)
        save_figures(results, predictions, summary)
        print_report(results.drop(columns="Experiment_Settings"), summary)
        print(f"Saved four backtesting CSVs and five figures under {output}")
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"Backtesting could not complete: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
