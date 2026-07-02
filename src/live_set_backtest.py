from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.config import settings
from src.live_set_db import load_current_set_live_rows
from src.live_set_features import (
    BASELINE_SET_FEATURE_COLUMNS,
    EXTENDED_SET_FEATURE_COLUMNS,
    EXTENDED_CLUTCH_STREAK_FEATURE_COLUMNS,
    STREAK_SET_FEATURE_COLUMNS,
    build_current_set_live_features,
)
from src.train import build_model


OUTPUT_DIR = Path("data/processed")
MIN_TOTAL_POINTS = 8
RALLY_STEP = 3


def time_based_split(rows: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    sorted_rows = rows.sort_values("snapshot_ts").reset_index(drop=True)
    split_index = int(len(sorted_rows) * (1 - test_size))

    if split_index <= 0 or split_index >= len(sorted_rows):
        raise ValueError("Not enough rows for a time-based split.")

    train_rows = sorted_rows.iloc[:split_index].copy()
    test_rows = sorted_rows.iloc[split_index:].copy()
    return train_rows, test_rows


def build_summary_frame(test_rows: pd.DataFrame, probabilities, predictions) -> pd.DataFrame:
    output = test_rows.reset_index(drop=True).copy()
    output["pred_set_team1_win_proba"] = probabilities
    output["pred_set_team1_win"] = predictions
    output["is_correct"] = (
        output["target_set_team1_win"] == output["pred_set_team1_win"]
    ).astype(int)
    return output


def run_experiment(
    rows: pd.DataFrame,
    label: str,
    feature_columns: list[str],
) -> tuple[dict[str, float | str], pd.DataFrame]:
    train_rows, test_rows = time_based_split(rows)
    combined_rows = pd.concat([train_rows, test_rows], ignore_index=True)

    x_all, y_all = build_current_set_live_features(combined_rows, feature_columns)
    train_count = len(train_rows)
    x_train = x_all.iloc[:train_count]
    y_train = y_all.iloc[:train_count]
    x_test = x_all.iloc[train_count:]
    y_test = y_all.iloc[train_count:]

    model = build_model(settings.model_family)
    model.fit(x_train, y_train)

    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    print(f"\n=== {label} ===")
    print(f"Feature columns: {', '.join(feature_columns)}")
    print(f"Loaded live-set rows: {len(rows)}")
    print(f"Training rows: {len(train_rows)}")
    print(f"Test rows: {len(test_rows)}")
    print(
        f"Train ts range: {train_rows['snapshot_ts'].min()} -> {train_rows['snapshot_ts'].max()}"
    )
    print(
        f"Test ts range: {test_rows['snapshot_ts'].min()} -> {test_rows['snapshot_ts'].max()}"
    )
    print(classification_report(y_test, predictions))

    accuracy = float(accuracy_score(y_test, predictions))
    f1_macro = float(f1_score(y_test, predictions, average="macro"))
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 macro: {f1_macro:.4f}")

    summary_frame = build_summary_frame(test_rows, probabilities, predictions)
    summary_frame["experiment"] = label
    return (
        {
            "experiment": label,
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "rows": len(rows),
            "train_rows": len(train_rows),
            "test_rows": len(test_rows),
        },
        summary_frame,
    )


def main() -> None:
    print(f"Current set live backtest model family: {settings.model_family}")
    print(f"Sampling filters: min_total_points={MIN_TOTAL_POINTS}, rally_step={RALLY_STEP}")

    rows = load_current_set_live_rows(
        min_total_points=MIN_TOTAL_POINTS,
        rally_step=RALLY_STEP,
    )
    baseline_result, baseline_frame = run_experiment(
        rows,
        "baseline",
        BASELINE_SET_FEATURE_COLUMNS,
    )
    extended_result, extended_frame = run_experiment(
        rows,
        "extended_clutch",
        EXTENDED_SET_FEATURE_COLUMNS,
    )
    streak_result, streak_frame = run_experiment(
        rows,
        "streak_4plus",
        STREAK_SET_FEATURE_COLUMNS,
    )
    clutch_streak_result, clutch_streak_frame = run_experiment(
        rows,
        "extended_clutch_streak_4plus",
        EXTENDED_CLUTCH_STREAK_FEATURE_COLUMNS,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "live_set_backtest_predictions.csv"
    summary_path = OUTPUT_DIR / "live_set_backtest_summary.csv"
    prediction_frame = pd.concat(
        [baseline_frame, extended_frame, streak_frame, clutch_streak_frame],
        ignore_index=True,
    )
    prediction_frame.to_csv(output_path, index=False)
    pd.DataFrame(
        [baseline_result, extended_result, streak_result, clutch_streak_result]
    ).to_csv(summary_path, index=False)
    print("\nComparison summary:")
    for result in [baseline_result, extended_result, streak_result, clutch_streak_result]:
        print(
            f"  {result['experiment']}: rows={result['rows']}, "
            f"accuracy={result['accuracy']:.4f}, f1_macro={result['f1_macro']:.4f}"
        )
    print(f"Predictions saved to {output_path}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
