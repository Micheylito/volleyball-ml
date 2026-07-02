from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/processed/live_set_backtest_predictions.csv")
OUTPUT_DIR = Path("data/processed")
DEFAULT_EXPERIMENT = "streak_4plus"
ODDS_RANGES = (
    (1.43, 1.50),
    (1.50, 1.60),
    (1.60, 1.70),
    (1.43, 1.60),
    (1.43, 1.70),
)
PROBABILITY_THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)


def prepare_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
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
    return df


def build_range_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    for min_odds, max_odds in ODDS_RANGES:
        range_label = f"{min_odds:.2f}-{max_odds:.2f}"
        range_rows = df[
            (df["selected_odds"] >= min_odds) & (df["selected_odds"] <= max_odds)
        ].copy()

        if range_rows.empty:
            for threshold in PROBABILITY_THRESHOLDS:
                rows.append(
                    {
                        "odds_range": range_label,
                        "threshold": threshold,
                        "signals": 0,
                        "coverage": 0.0,
                        "accuracy": 0.0,
                        "avg_odds": 0.0,
                    }
                )
            continue

        range_size = len(range_rows)
        for threshold in PROBABILITY_THRESHOLDS:
            selected = range_rows[range_rows["selected_probability"] >= threshold].copy()
            if selected.empty:
                rows.append(
                    {
                        "odds_range": range_label,
                        "threshold": threshold,
                        "signals": 0,
                        "coverage": 0.0,
                        "accuracy": 0.0,
                        "avg_odds": 0.0,
                    }
                )
                continue

            rows.append(
                {
                    "odds_range": range_label,
                    "threshold": threshold,
                    "signals": int(len(selected)),
                    "coverage": float(len(selected) / range_size),
                    "accuracy": float(selected["is_correct"].mean()),
                    "avg_odds": float(selected["selected_odds"].mean()),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"{INPUT_PATH} not found. Run python -m src.live_set_backtest first."
        )

    predictions = pd.read_csv(INPUT_PATH)
    prepared = prepare_predictions(predictions)
    summary = build_range_summary(prepared)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "live_set_range_summary.csv"
    summary.to_csv(output_path, index=False)

    print("Live set range scan")
    print(f"Experiment: {DEFAULT_EXPERIMENT}")
    print("\nRange / threshold summary:")
    for odds_range in summary["odds_range"].drop_duplicates().tolist():
        range_rows = summary[summary["odds_range"] == odds_range].copy()
        print(f"\nOdds range {odds_range}:")
        for row in range_rows.itertuples(index=False):
            print(
                f"  proba>={row.threshold:.2f}: "
                f"signals={row.signals}, coverage={row.coverage:.4f}, "
                f"accuracy={row.accuracy:.4f}, avg_odds={row.avg_odds:.2f}"
            )

    print(f"\nRange summary saved to {output_path}")


if __name__ == "__main__":
    main()
