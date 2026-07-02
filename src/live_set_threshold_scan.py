from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/processed/live_set_backtest_predictions.csv")
OUTPUT_DIR = Path("data/processed")
DEFAULT_EXPERIMENT = "streak_4plus"
MIN_ODDS = 1.43
PROBABILITY_THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)


def prepare_filtered_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    df = predictions[predictions["experiment"] == DEFAULT_EXPERIMENT].copy()
    if df.empty:
        raise ValueError(f"No rows found for experiment={DEFAULT_EXPERIMENT}.")

    df["selected_side"] = df["pred_set_team1_win"].map({1: "home", 0: "away"})
    df["selected_probability"] = df["pred_set_team1_win_proba"]
    df["selected_odds"] = df["set_win1"]

    away_mask = df["pred_set_team1_win"] == 0
    df.loc[away_mask, "selected_probability"] = 1.0 - df.loc[
        away_mask, "pred_set_team1_win_proba"
    ]
    df.loc[away_mask, "selected_odds"] = df.loc[away_mask, "set_win2"]

    filtered = df[df["selected_odds"] >= MIN_ODDS].copy()
    if filtered.empty:
        raise ValueError(f"No rows found with selected_odds >= {MIN_ODDS:.2f}.")

    return filtered


def build_threshold_summary(filtered: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for threshold in PROBABILITY_THRESHOLDS:
        selected = filtered[filtered["selected_probability"] >= threshold].copy()
        if selected.empty:
            rows.append(
                {
                    "threshold": threshold,
                    "signals": 0,
                    "coverage": 0.0,
                    "accuracy": 0.0,
                    "avg_odds": 0.0,
                    "min_odds": 0.0,
                    "max_odds": 0.0,
                }
            )
            continue

        rows.append(
            {
                "threshold": threshold,
                "signals": int(len(selected)),
                "coverage": float(len(selected) / len(filtered)),
                "accuracy": float(selected["is_correct"].mean()),
                "avg_odds": float(selected["selected_odds"].mean()),
                "min_odds": float(selected["selected_odds"].min()),
                "max_odds": float(selected["selected_odds"].max()),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"{INPUT_PATH} not found. Run python -m src.live_set_backtest first."
        )

    predictions = pd.read_csv(INPUT_PATH)
    filtered = prepare_filtered_predictions(predictions)
    summary = build_threshold_summary(filtered)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "live_set_threshold_summary.csv"
    summary.to_csv(output_path, index=False)

    print("Live set threshold scan")
    print(f"Experiment: {DEFAULT_EXPERIMENT}")
    print(f"Zone: selected_odds >= {MIN_ODDS:.2f}")
    print(f"Rows in zone: {len(filtered)}")
    print(f"Base accuracy in zone: {filtered['is_correct'].mean():.4f}")
    print("\nProbability threshold summary:")
    for row in summary.itertuples(index=False):
        print(
            f"  proba>={row.threshold:.2f}: "
            f"signals={row.signals}, coverage={row.coverage:.4f}, "
            f"accuracy={row.accuracy:.4f}, avg_odds={row.avg_odds:.2f}, "
            f"range={row.min_odds:.2f}-{row.max_odds:.2f}"
        )
    print(f"\nThreshold summary saved to {output_path}")


if __name__ == "__main__":
    main()
