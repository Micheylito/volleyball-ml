from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import settings
from src.db import load_matches, load_rally_backtest_snapshots
from src.features import build_features, build_inference_features
from src.train import build_model, time_based_split


OUTPUT_DIR = Path("data/processed")
SIGNAL_THRESHOLD = 0.80


def train_reference_model(train_matches: pd.DataFrame):
    x_train, y_train = build_features(train_matches, active_blocks=settings.feature_blocks)
    model = build_model(settings.model_family)
    model.fit(x_train, y_train)
    return model


def build_signal_rows(
    model,
    train_matches: pd.DataFrame,
    rally_snapshots: pd.DataFrame,
) -> pd.DataFrame:
    if rally_snapshots.empty:
        return rally_snapshots.copy()

    combined = pd.concat([train_matches, rally_snapshots], ignore_index=True)
    inference_features = build_inference_features(combined, active_blocks=settings.feature_blocks)
    snapshot_features = inference_features.tail(len(rally_snapshots)).reset_index(drop=True)

    probabilities = model.predict_proba(snapshot_features)[:, 1]
    output = rally_snapshots.reset_index(drop=True).copy()
    output["pred_home_win_proba"] = probabilities
    output["signal_side"] = pd.NA
    output.loc[output["pred_home_win_proba"] >= SIGNAL_THRESHOLD, "signal_side"] = "home"
    output.loc[output["pred_home_win_proba"] <= (1.0 - SIGNAL_THRESHOLD), "signal_side"] = "away"
    output = output[output["signal_side"].notna()].copy()

    if output.empty:
        return output

    output["signal_probability"] = output["pred_home_win_proba"]
    away_mask = output["signal_side"] == "away"
    output.loc[away_mask, "signal_probability"] = 1.0 - output.loc[away_mask, "pred_home_win_proba"]
    output["market_odds"] = output["home_odds"]
    output.loc[away_mask, "market_odds"] = output.loc[away_mask, "away_odds"]
    output["is_correct"] = 0
    output.loc[
        (output["signal_side"] == "home") & (output["actual_winner"] == 1),
        "is_correct",
    ] = 1
    output.loc[
        (output["signal_side"] == "away") & (output["actual_winner"] == 2),
        "is_correct",
    ] = 1
    output["snapshot_ts"] = pd.to_datetime(output["match_date"])
    return output.sort_values(
        ["snapshot_ts", "match_id", "set_number", "rally_number"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)


def summarize_signals(signals: pd.DataFrame, summary_name: str) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame(
            columns=[
                "summary_name",
                "signal_side",
                "signal_rows",
                "unique_matches",
                "accuracy",
                "avg_probability",
                "avg_market_odds",
                "median_market_odds",
                "p10_market_odds",
                "p90_market_odds",
                "min_market_odds",
                "max_market_odds",
                "avg_set_number",
            ]
        )

    grouped = (
        signals.groupby("signal_side", dropna=False)
        .agg(
            signal_rows=("match_id", "count"),
            unique_matches=("match_id", "nunique"),
            accuracy=("is_correct", "mean"),
            avg_probability=("signal_probability", "mean"),
            avg_market_odds=("market_odds", "mean"),
            median_market_odds=("market_odds", "median"),
            p10_market_odds=("market_odds", lambda values: float(values.quantile(0.10))),
            p90_market_odds=("market_odds", lambda values: float(values.quantile(0.90))),
            min_market_odds=("market_odds", "min"),
            max_market_odds=("market_odds", "max"),
            avg_set_number=("set_number", "mean"),
        )
        .reset_index()
    )
    grouped.insert(0, "summary_name", summary_name)
    return grouped


def main() -> None:
    print(f"Rally backtest model family: {settings.model_family}")
    print(f"Rally backtest feature blocks: {', '.join(settings.feature_blocks)}")
    print(f"Signal threshold: {SIGNAL_THRESHOLD:.0%}")

    matches = load_matches()
    train_matches, test_matches = time_based_split(matches)
    model = train_reference_model(train_matches)

    test_match_ids = test_matches["match_id"].astype(int).tolist()
    rally_snapshots = load_rally_backtest_snapshots(test_match_ids)
    print(f"Train matches: {len(train_matches)}")
    print(f"Test matches: {len(test_matches)}")
    print(f"Loaded rally snapshots: {len(rally_snapshots)}")

    signal_rows = build_signal_rows(model, train_matches, rally_snapshots)
    print(f"Signal rows above threshold: {len(signal_rows)}")

    first_signal_rows = (
        signal_rows.sort_values(["snapshot_ts", "match_id", "set_number", "rally_number"])
        .drop_duplicates(subset=["match_id"], keep="first")
        .reset_index(drop=True)
    )

    summary_all = summarize_signals(signal_rows, "all_signal_rows")
    summary_first = summarize_signals(first_signal_rows, "first_signal_per_match")
    combined_summary = pd.concat([summary_all, summary_first], ignore_index=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    signal_rows_path = OUTPUT_DIR / "rally_backtest_signal_rows.csv"
    first_signal_rows_path = OUTPUT_DIR / "rally_backtest_first_signal_rows.csv"
    summary_path = OUTPUT_DIR / "rally_backtest_signal_summary.csv"

    signal_rows.to_csv(signal_rows_path, index=False)
    first_signal_rows.to_csv(first_signal_rows_path, index=False)
    combined_summary.to_csv(summary_path, index=False)

    print("Rally signal summary:")
    if combined_summary.empty:
        print("  No signals found above threshold.")
    else:
        for row in combined_summary.itertuples(index=False):
            print(
                f"  {row.summary_name} | {row.signal_side}: "
                f"rows={row.signal_rows}, matches={row.unique_matches}, "
                f"accuracy={row.accuracy:.4f}, avg_odds={row.avg_market_odds:.2f}, "
                f"median_odds={row.median_market_odds:.2f}, "
                f"p10={row.p10_market_odds:.2f}, p90={row.p90_market_odds:.2f}, "
                f"range={row.min_market_odds:.2f}-{row.max_market_odds:.2f}"
            )

    print(f"Signal rows saved to {signal_rows_path}")
    print(f"First signal rows saved to {first_signal_rows_path}")
    print(f"Signal summary saved to {summary_path}")


if __name__ == "__main__":
    main()
