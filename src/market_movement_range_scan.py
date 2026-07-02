from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.market_movement_analysis import prepare_market_movement_rows


OUTPUT_DIR = Path("data/processed")
ODDS_RANGES = (
    (1.43, 1.50),
    (1.50, 1.60),
    (1.60, 1.70),
    (1.43, 1.60),
    (1.43, 1.70),
)


def prepare_scan_rows() -> pd.DataFrame:
    rows = prepare_market_movement_rows().copy()
    rows["live_favorite_odds"] = pd.to_numeric(rows["match_win1"], errors="coerce")
    away_mask = rows["live_favorite"] == "away"
    rows.loc[away_mask, "live_favorite_odds"] = pd.to_numeric(
        rows.loc[away_mask, "match_win2"], errors="coerce"
    )

    rows["state"] = "no_flip"
    rows.loc[
        (rows["favorite_flipped"] == 1) & (rows["live_favorite"] == "away"),
        "state",
    ] = "flipped_to_away"
    rows.loc[
        (rows["favorite_flipped"] == 1) & (rows["live_favorite"] == "home"),
        "state",
    ] = "flipped_to_home"

    first_flips = (
        rows[rows["favorite_flipped"] == 1]
        .sort_values(["snapshot_ts", "match_id", "set_number", "rally_number"])
        .drop_duplicates(subset=["match_id"], keep="first")
        .copy()
    )
    rows["is_first_flip_per_match"] = 0
    if not first_flips.empty:
        first_flip_keys = set(
            zip(
                first_flips["match_id"].tolist(),
                first_flips["set_number"].tolist(),
                first_flips["rally_number"].tolist(),
            )
        )
        rows["is_first_flip_per_match"] = [
            int((match_id, set_number, rally_number) in first_flip_keys)
            for match_id, set_number, rally_number in zip(
                rows["match_id"], rows["set_number"], rows["rally_number"]
            )
        ]

    return rows


def summarize_range_state(rows: pd.DataFrame, min_odds: float, max_odds: float, state: str) -> dict[str, float | int | str]:
    scoped = rows[
        (rows["live_favorite_odds"] >= min_odds)
        & (rows["live_favorite_odds"] <= max_odds)
        & (rows["state"] == state)
    ].copy()
    if scoped.empty:
        return {
            "odds_range": f"{min_odds:.2f}-{max_odds:.2f}",
            "state": state,
            "snapshots": 0,
            "matches": 0,
            "live_favorite_accuracy": 0.0,
            "reference_favorite_accuracy": 0.0,
            "avg_set_number": 0.0,
            "avg_shift": 0.0,
            "avg_live_favorite_odds": 0.0,
        }

    return {
        "odds_range": f"{min_odds:.2f}-{max_odds:.2f}",
        "state": state,
        "snapshots": int(len(scoped)),
        "matches": int(scoped["match_id"].nunique()),
        "live_favorite_accuracy": float(scoped["live_favorite_correct"].mean()),
        "reference_favorite_accuracy": float(scoped["reference_favorite_correct"].mean()),
        "avg_set_number": float(scoped["set_number"].mean()),
        "avg_shift": float(scoped["home_prob_shift"].mean()),
        "avg_live_favorite_odds": float(scoped["live_favorite_odds"].mean()),
    }


def summarize_first_flip_ranges(rows: pd.DataFrame, min_odds: float, max_odds: float) -> dict[str, float | int | str]:
    scoped = rows[
        (rows["live_favorite_odds"] >= min_odds)
        & (rows["live_favorite_odds"] <= max_odds)
        & (rows["is_first_flip_per_match"] == 1)
    ].copy()
    if scoped.empty:
        return {
            "odds_range": f"{min_odds:.2f}-{max_odds:.2f}",
            "state": "first_flip_per_match",
            "snapshots": 0,
            "matches": 0,
            "live_favorite_accuracy": 0.0,
            "reference_favorite_accuracy": 0.0,
            "avg_set_number": 0.0,
            "avg_shift": 0.0,
            "avg_live_favorite_odds": 0.0,
        }

    return {
        "odds_range": f"{min_odds:.2f}-{max_odds:.2f}",
        "state": "first_flip_per_match",
        "snapshots": int(len(scoped)),
        "matches": int(scoped["match_id"].nunique()),
        "live_favorite_accuracy": float(scoped["live_favorite_correct"].mean()),
        "reference_favorite_accuracy": float(scoped["reference_favorite_correct"].mean()),
        "avg_set_number": float(scoped["set_number"].mean()),
        "avg_shift": float(scoped["home_prob_shift"].mean()),
        "avg_live_favorite_odds": float(scoped["live_favorite_odds"].mean()),
    }


def build_summary(rows: pd.DataFrame) -> pd.DataFrame:
    states = ("no_flip", "flipped_to_away", "flipped_to_home")
    records: list[dict[str, float | int | str]] = []
    for min_odds, max_odds in ODDS_RANGES:
        for state in states:
            records.append(summarize_range_state(rows, min_odds, max_odds, state))
        records.append(summarize_first_flip_ranges(rows, min_odds, max_odds))
    return pd.DataFrame(records)


def main() -> None:
    rows = prepare_scan_rows()
    summary = build_summary(rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows_path = OUTPUT_DIR / "market_movement_range_rows.csv"
    summary_path = OUTPUT_DIR / "market_movement_range_summary.csv"
    rows.to_csv(rows_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("Market movement range scan")
    for odds_range in summary["odds_range"].drop_duplicates().tolist():
        print(f"\nOdds range {odds_range}:")
        range_rows = summary[summary["odds_range"] == odds_range]
        for row in range_rows.itertuples(index=False):
            print(
                f"  {row.state}: snapshots={row.snapshots}, matches={row.matches}, "
                f"live_fav_acc={row.live_favorite_accuracy:.4f}, "
                f"ref_fav_acc={row.reference_favorite_accuracy:.4f}, "
                f"avg_odds={row.avg_live_favorite_odds:.2f}, "
                f"avg_shift={row.avg_shift:.4f}, avg_set={row.avg_set_number:.2f}"
            )

    print(f"\nRows saved to {rows_path}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
