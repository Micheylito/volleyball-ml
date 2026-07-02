from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/processed/live_set_backtest_predictions.csv")
OUTPUT_DIR = Path("data/processed")
DEFAULT_EXPERIMENT = "streak_4plus"
MIN_ODDS = 1.43
MIN_SIGNALS = 40
MIN_ACCURACY = 0.80

PATTERN_COLUMNS = [
    "selected_side",
    "set_number",
    "phase_bucket",
    "score_gap_bucket",
    "selected_side_is_leading",
    "selected_side_is_favorite",
    "selected_side_streak_bucket",
]


def _phase_bucket(total_points: float) -> str:
    if total_points <= 15:
        return "early_8_15"
    if total_points <= 21:
        return "mid_16_21"
    if total_points <= 27:
        return "late_22_27"
    return "extended_28_plus"


def _score_gap_bucket(score_gap_abs: float) -> str:
    if score_gap_abs == 0:
        return "gap_0"
    if score_gap_abs == 1:
        return "gap_1"
    if score_gap_abs == 2:
        return "gap_2"
    if score_gap_abs <= 4:
        return "gap_3_4"
    return "gap_5_plus"


def _streak_bucket(value: float) -> str:
    if value <= 1:
        return "streak_0_1"
    if value <= 3:
        return "streak_2_3"
    return "streak_4_plus"


def infer_point_streaks(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched = enriched.sort_values(
        ["match_id", "set_number", "rally_number", "snapshot_ts"]
    ).reset_index(drop=True)

    team1_streaks: list[int] = []
    team2_streaks: list[int] = []
    current_group: tuple[int, int] | None = None
    current_winner = 0
    current_streak = 0
    previous_score1: float | None = None
    previous_score2: float | None = None

    for row in enriched.itertuples(index=False):
        group_key = (int(row.match_id), int(row.set_number))
        score1 = float(row.score1)
        score2 = float(row.score2)

        if group_key != current_group:
            current_group = group_key
            current_winner = 0
            current_streak = 0
            previous_score1 = None
            previous_score2 = None

        point_winner = 0
        if previous_score1 is not None and previous_score2 is not None:
            delta1 = score1 - previous_score1
            delta2 = score2 - previous_score2
            if delta1 == 1 and delta2 == 0:
                point_winner = 1
            elif delta1 == 0 and delta2 == 1:
                point_winner = 2

        if point_winner in (1, 2):
            if point_winner == current_winner:
                current_streak += 1
            else:
                current_winner = point_winner
                current_streak = 1
        else:
            current_winner = 0
            current_streak = 0

        if current_winner == 1:
            team1_streaks.append(current_streak)
            team2_streaks.append(0)
        elif current_winner == 2:
            team1_streaks.append(0)
            team2_streaks.append(current_streak)
        else:
            team1_streaks.append(0)
            team2_streaks.append(0)

        previous_score1 = score1
        previous_score2 = score2

    enriched["team1_point_streak_inferred"] = team1_streaks
    enriched["team2_point_streak_inferred"] = team2_streaks
    return enriched


def prepare_patterns(df: pd.DataFrame) -> pd.DataFrame:
    prepared = infer_point_streaks(df)
    prepared["selected_side"] = prepared["pred_set_team1_win"].map({1: "home", 0: "away"})
    prepared["selected_odds"] = prepared["set_win1"]
    away_mask = prepared["pred_set_team1_win"] == 0
    prepared.loc[away_mask, "selected_odds"] = prepared.loc[away_mask, "set_win2"]
    prepared["selected_probability"] = prepared["pred_set_team1_win_proba"]
    prepared.loc[away_mask, "selected_probability"] = (
        1.0 - prepared.loc[away_mask, "pred_set_team1_win_proba"]
    )

    prepared["score_gap_abs"] = (prepared["score1"] - prepared["score2"]).abs()
    prepared["set_total_points"] = prepared["score1"] + prepared["score2"]
    prepared["phase_bucket"] = prepared["set_total_points"].map(_phase_bucket)
    prepared["score_gap_bucket"] = prepared["score_gap_abs"].map(_score_gap_bucket)

    prepared["selected_side_is_leading"] = "no"
    prepared.loc[
        ((prepared["pred_set_team1_win"] == 1) & (prepared["score1"] > prepared["score2"]))
        | ((prepared["pred_set_team1_win"] == 0) & (prepared["score2"] > prepared["score1"])),
        "selected_side_is_leading",
    ] = "yes"
    prepared.loc[prepared["score1"] == prepared["score2"], "selected_side_is_leading"] = "tie"

    prepared["selected_side_is_favorite"] = "no"
    prepared.loc[
        ((prepared["pred_set_team1_win"] == 1) & (prepared["set_win1"] <= prepared["set_win2"]))
        | ((prepared["pred_set_team1_win"] == 0) & (prepared["set_win2"] <= prepared["set_win1"])),
        "selected_side_is_favorite",
    ] = "yes"

    prepared["selected_side_streak"] = prepared["team1_point_streak_inferred"]
    prepared.loc[away_mask, "selected_side_streak"] = prepared.loc[
        away_mask, "team2_point_streak_inferred"
    ]
    prepared["selected_side_streak_bucket"] = prepared["selected_side_streak"].map(_streak_bucket)
    prepared["set_number"] = pd.to_numeric(prepared["set_number"], errors="coerce").fillna(0).astype(int)

    prepared = prepared[prepared["selected_odds"] >= MIN_ODDS].copy()
    return prepared


def summarize_pattern_group(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    grouped = (
        df.groupby(columns, dropna=False)
        .agg(
            signals=("is_correct", "count"),
            accuracy=("is_correct", "mean"),
            avg_odds=("selected_odds", "mean"),
            avg_probability=("selected_probability", "mean"),
        )
        .reset_index()
    )
    grouped["pattern_keys"] = ", ".join(columns)
    grouped["pattern_values"] = grouped[columns].astype(str).agg(" | ".join, axis=1)
    return grouped[
        ["pattern_keys", "pattern_values", "signals", "accuracy", "avg_odds", "avg_probability"]
    ]


def build_pattern_summary(df: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for size in (1, 2, 3):
        for columns in combinations(PATTERN_COLUMNS, size):
            frames.append(summarize_pattern_group(df, list(columns)))

    summary = pd.concat(frames, ignore_index=True)
    summary = summary[
        (summary["signals"] >= MIN_SIGNALS) & (summary["accuracy"] >= MIN_ACCURACY)
    ].copy()
    summary = summary.sort_values(
        ["accuracy", "signals", "avg_odds"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return summary


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"{INPUT_PATH} not found. Run python -m src.live_set_backtest first."
        )

    predictions = pd.read_csv(INPUT_PATH)
    predictions["snapshot_ts"] = pd.to_datetime(predictions["snapshot_ts"])
    experiment_rows = predictions[predictions["experiment"] == DEFAULT_EXPERIMENT].copy()
    if experiment_rows.empty:
        raise ValueError(f"No rows found for experiment={DEFAULT_EXPERIMENT}.")

    prepared = prepare_patterns(experiment_rows)
    if prepared.empty:
        raise ValueError(
            f"No rows left after filtering experiment={DEFAULT_EXPERIMENT} and odds>={MIN_ODDS}."
        )

    pattern_summary = build_pattern_summary(prepared)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "live_set_pattern_summary.csv"
    pattern_summary.to_csv(output_path, index=False)

    overall_accuracy = float(prepared["is_correct"].mean())
    print("Live set pattern scan")
    print(f"Experiment: {DEFAULT_EXPERIMENT}")
    print(f"Rows with selected_odds >= {MIN_ODDS:.2f}: {len(prepared)}")
    print(f"Overall accuracy in filtered zone: {overall_accuracy:.4f}")
    print(
        f"Strong patterns: accuracy >= {MIN_ACCURACY:.2%}, signals >= {MIN_SIGNALS}"
    )
    if pattern_summary.empty:
        print("No strong patterns found with current thresholds.")
    else:
        for row in pattern_summary.head(20).itertuples(index=False):
            print(
                f"  {row.pattern_keys} => {row.pattern_values}: "
                f"signals={row.signals}, accuracy={row.accuracy:.4f}, "
                f"avg_odds={row.avg_odds:.2f}, avg_proba={row.avg_probability:.4f}"
            )
    print(f"Pattern summary saved to {output_path}")


if __name__ == "__main__":
    main()
