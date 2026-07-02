from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.db import load_matches
from src.live_set_db import load_current_set_live_rows


OUTPUT_DIR = Path("data/processed")
RALLY_STEP = 3
MIN_TOTAL_POINTS = 8


def _normalized_home_probability(home_odds: pd.Series, away_odds: pd.Series) -> pd.Series:
    implied_home = 1.0 / pd.to_numeric(home_odds, errors="coerce")
    implied_away = 1.0 / pd.to_numeric(away_odds, errors="coerce")
    overround = implied_home + implied_away
    return implied_home / overround


def _shift_bucket(value: float) -> str:
    if value <= -0.20:
        return "home_big_drift"
    if value <= -0.10:
        return "home_medium_drift"
    if value <= -0.03:
        return "home_small_drift"
    if value < 0.03:
        return "near_reference"
    if value < 0.10:
        return "home_small_steam"
    if value < 0.20:
        return "home_medium_steam"
    return "home_big_steam"


def prepare_market_movement_rows() -> pd.DataFrame:
    matches = load_matches()
    live_rows = load_current_set_live_rows(
        min_total_points=MIN_TOTAL_POINTS,
        rally_step=RALLY_STEP,
    )

    reference_columns = matches[
        ["match_id", "home_odds", "away_odds", "winner", "odds_source"]
    ].rename(
        columns={
            "home_odds": "reference_home_odds",
            "away_odds": "reference_away_odds",
            "winner": "match_winner",
            "odds_source": "reference_odds_source",
        }
    )

    merged = live_rows.merge(reference_columns, on="match_id", how="inner")
    merged["snapshot_ts"] = pd.to_datetime(merged["snapshot_ts"])
    merged["set_number"] = pd.to_numeric(merged["set_number"], errors="coerce").fillna(0).astype(int)

    merged["current_home_prob"] = _normalized_home_probability(
        merged["match_win1"], merged["match_win2"]
    )
    merged["reference_home_prob"] = _normalized_home_probability(
        merged["reference_home_odds"], merged["reference_away_odds"]
    )
    merged["home_prob_shift"] = (
        merged["current_home_prob"] - merged["reference_home_prob"]
    ).fillna(0.0)
    merged["shift_bucket"] = merged["home_prob_shift"].map(_shift_bucket)

    merged["reference_favorite"] = "home"
    merged.loc[
        pd.to_numeric(merged["reference_away_odds"], errors="coerce")
        < pd.to_numeric(merged["reference_home_odds"], errors="coerce"),
        "reference_favorite",
    ] = "away"

    merged["live_favorite"] = "home"
    merged.loc[
        pd.to_numeric(merged["match_win2"], errors="coerce")
        < pd.to_numeric(merged["match_win1"], errors="coerce"),
        "live_favorite",
    ] = "away"

    merged["favorite_flipped"] = (merged["reference_favorite"] != merged["live_favorite"]).astype(int)
    merged["live_favorite_correct"] = 0
    merged.loc[(merged["live_favorite"] == "home") & (merged["match_winner"] == 1), "live_favorite_correct"] = 1
    merged.loc[(merged["live_favorite"] == "away") & (merged["match_winner"] == 2), "live_favorite_correct"] = 1

    merged["reference_favorite_correct"] = 0
    merged.loc[
        (merged["reference_favorite"] == "home") & (merged["match_winner"] == 1),
        "reference_favorite_correct",
    ] = 1
    merged.loc[
        (merged["reference_favorite"] == "away") & (merged["match_winner"] == 2),
        "reference_favorite_correct",
    ] = 1

    return merged


def summarize_shift_buckets(rows: pd.DataFrame) -> pd.DataFrame:
    summary = (
        rows.groupby("shift_bucket", dropna=False)
        .agg(
            snapshots=("match_id", "count"),
            matches=("match_id", "nunique"),
            live_favorite_accuracy=("live_favorite_correct", "mean"),
            reference_favorite_accuracy=("reference_favorite_correct", "mean"),
            flip_rate=("favorite_flipped", "mean"),
            avg_set_number=("set_number", "mean"),
            avg_shift=("home_prob_shift", "mean"),
        )
        .reset_index()
    )
    return summary.sort_values("avg_shift").reset_index(drop=True)


def summarize_flip_states(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        rows.groupby(["favorite_flipped", "live_favorite"], dropna=False)
        .agg(
            snapshots=("match_id", "count"),
            matches=("match_id", "nunique"),
            live_favorite_accuracy=("live_favorite_correct", "mean"),
            reference_favorite_accuracy=("reference_favorite_correct", "mean"),
            avg_shift=("home_prob_shift", "mean"),
            avg_set_number=("set_number", "mean"),
        )
        .reset_index()
    )
    grouped["flip_state"] = grouped["favorite_flipped"].map({0: "no_flip", 1: "flipped"})
    return grouped[
        [
            "flip_state",
            "live_favorite",
            "snapshots",
            "matches",
            "live_favorite_accuracy",
            "reference_favorite_accuracy",
            "avg_shift",
            "avg_set_number",
        ]
    ].sort_values(["flip_state", "live_favorite"]).reset_index(drop=True)


def summarize_first_flip_per_match(rows: pd.DataFrame) -> pd.DataFrame:
    flips = rows[rows["favorite_flipped"] == 1].copy()
    if flips.empty:
        return pd.DataFrame(
            columns=[
                "group",
                "matches",
                "live_favorite_accuracy",
                "reference_favorite_accuracy",
                "avg_shift",
                "avg_set_number",
            ]
        )

    first_flips = (
        flips.sort_values(["snapshot_ts", "match_id", "set_number", "rally_number"])
        .drop_duplicates(subset=["match_id"], keep="first")
        .reset_index(drop=True)
    )
    summary = pd.DataFrame(
        [
            {
                "group": "first_flip_per_match",
                "matches": int(first_flips["match_id"].nunique()),
                "live_favorite_accuracy": float(first_flips["live_favorite_correct"].mean()),
                "reference_favorite_accuracy": float(first_flips["reference_favorite_correct"].mean()),
                "avg_shift": float(first_flips["home_prob_shift"].mean()),
                "avg_set_number": float(first_flips["set_number"].mean()),
            }
        ]
    )
    return summary


def main() -> None:
    rows = prepare_market_movement_rows()
    shift_summary = summarize_shift_buckets(rows)
    flip_summary = summarize_flip_states(rows)
    first_flip_summary = summarize_first_flip_per_match(rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows_path = OUTPUT_DIR / "market_movement_rows.csv"
    shift_path = OUTPUT_DIR / "market_movement_shift_summary.csv"
    flip_path = OUTPUT_DIR / "market_movement_flip_summary.csv"
    first_flip_path = OUTPUT_DIR / "market_movement_first_flip_summary.csv"

    rows.to_csv(rows_path, index=False)
    shift_summary.to_csv(shift_path, index=False)
    flip_summary.to_csv(flip_path, index=False)
    first_flip_summary.to_csv(first_flip_path, index=False)

    print("Market movement analysis")
    print(f"Sampling filters: min_total_points={MIN_TOTAL_POINTS}, rally_step={RALLY_STEP}")
    print(f"Snapshots: {len(rows)}")
    print(f"Matches: {rows['match_id'].nunique()}")
    print("\nShift bucket summary:")
    for row in shift_summary.itertuples(index=False):
        print(
            f"  {row.shift_bucket}: snapshots={row.snapshots}, matches={row.matches}, "
            f"live_fav_acc={row.live_favorite_accuracy:.4f}, "
            f"ref_fav_acc={row.reference_favorite_accuracy:.4f}, "
            f"flip_rate={row.flip_rate:.4f}, avg_shift={row.avg_shift:.4f}, "
            f"avg_set={row.avg_set_number:.2f}"
        )

    print("\nFlip state summary:")
    for row in flip_summary.itertuples(index=False):
        print(
            f"  {row.flip_state} | live_favorite={row.live_favorite}: "
            f"snapshots={row.snapshots}, matches={row.matches}, "
            f"live_fav_acc={row.live_favorite_accuracy:.4f}, "
            f"ref_fav_acc={row.reference_favorite_accuracy:.4f}, "
            f"avg_shift={row.avg_shift:.4f}, avg_set={row.avg_set_number:.2f}"
        )

    if not first_flip_summary.empty:
        row = first_flip_summary.iloc[0]
        print("\nFirst flip per match:")
        print(
            f"  matches={int(row['matches'])}, "
            f"live_fav_acc={row['live_favorite_accuracy']:.4f}, "
            f"ref_fav_acc={row['reference_favorite_accuracy']:.4f}, "
            f"avg_shift={row['avg_shift']:.4f}, avg_set={row['avg_set_number']:.2f}"
        )

    print(f"\nRows saved to {rows_path}")
    print(f"Shift summary saved to {shift_path}")
    print(f"Flip summary saved to {flip_path}")
    print(f"First-flip summary saved to {first_flip_path}")


if __name__ == "__main__":
    main()
