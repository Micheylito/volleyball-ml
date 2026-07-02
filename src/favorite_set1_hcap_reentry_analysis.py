from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.db import load_favorite_set1_serve_rows
from src.favorite_set1_serve_analysis import prepare_rows
from src.favorite_set1_set2_side_markets_analysis import load_set2_final_scores
from src.live_set_db import load_set2_hcap_trajectory_rows


OUTPUT_DIR = Path("data/processed")
TARGET_HCAP_LINE = -2.5
TARGET_FAVORITE_ODDS_MAX = 1.50
TARGET_SERVE_GAP_MIN = 0.08


def build_base_matches() -> pd.DataFrame:
    rows = prepare_rows(load_favorite_set1_serve_rows()).copy()
    rows = rows[rows["favorite_won_set1"] == 1].copy()

    rows["favorite_pre_match_odds"] = rows["home_odds"]
    away_favorite_mask = rows["favorite_team"] == 2
    rows.loc[away_favorite_mask, "favorite_pre_match_odds"] = rows.loc[
        away_favorite_mask, "away_odds"
    ]

    rows["favorite_pre_match_odds"] = pd.to_numeric(
        rows["favorite_pre_match_odds"], errors="coerce"
    )
    rows["favorite_set1_serve_gap"] = pd.to_numeric(
        rows["favorite_set1_serve_gap"], errors="coerce"
    )

    rows = rows[
        rows["favorite_pre_match_odds"].notna()
        & (rows["favorite_pre_match_odds"] <= TARGET_FAVORITE_ODDS_MAX)
        & rows["favorite_set1_serve_gap"].notna()
        & (rows["favorite_set1_serve_gap"] >= TARGET_SERVE_GAP_MIN)
    ].copy()
    return rows


def build_trajectory_rows(base_matches: pd.DataFrame) -> pd.DataFrame:
    trajectory = load_set2_hcap_trajectory_rows().copy()
    trajectory["snapshot_ts"] = pd.to_datetime(trajectory["snapshot_ts"])
    for column in [
        "rally_number",
        "score1",
        "score2",
        "set_hcap_line",
        "set_hcap1",
        "set_hcap2",
    ]:
        trajectory[column] = pd.to_numeric(trajectory[column], errors="coerce")

    merged = base_matches.merge(
        trajectory,
        on="match_id",
        how="inner",
        suffixes=("", "_traj"),
    )

    merged["favorite_set2_hcap_line"] = merged["set_hcap_line"]
    merged["favorite_set2_hcap_odds"] = merged["set_hcap1"]
    merged["favorite_score_at_entry"] = merged["score1"]
    merged["opponent_score_at_entry"] = merged["score2"]

    away_favorite_mask = merged["favorite_team"] == 2
    merged.loc[away_favorite_mask, "favorite_set2_hcap_line"] = -merged.loc[
        away_favorite_mask, "set_hcap_line"
    ]
    merged.loc[away_favorite_mask, "favorite_set2_hcap_odds"] = merged.loc[
        away_favorite_mask, "set_hcap2"
    ]
    merged.loc[away_favorite_mask, "favorite_score_at_entry"] = merged.loc[
        away_favorite_mask, "score2"
    ]
    merged.loc[away_favorite_mask, "opponent_score_at_entry"] = merged.loc[
        away_favorite_mask, "score1"
    ]

    merged["favorite_score_gap_at_entry"] = (
        merged["favorite_score_at_entry"] - merged["opponent_score_at_entry"]
    )
    merged["is_target_line"] = (
        merged["favorite_set2_hcap_line"].round(2) == TARGET_HCAP_LINE
    ).astype(int)
    return merged


def attach_final_outcome(rows: pd.DataFrame) -> pd.DataFrame:
    set2_scores = load_set2_final_scores().copy()
    for column in ["set2_score1", "set2_score2"]:
        set2_scores[column] = pd.to_numeric(set2_scores[column], errors="coerce")

    merged = rows.merge(set2_scores, on="match_id", how="inner")
    merged["favorite_set2_cover_target"] = (
        (merged["set2_score1"] + TARGET_HCAP_LINE) > merged["set2_score2"]
    ).astype(int)

    away_favorite_mask = merged["favorite_team"] == 2
    merged.loc[away_favorite_mask, "favorite_set2_cover_target"] = (
        (
            merged.loc[away_favorite_mask, "set2_score2"] + TARGET_HCAP_LINE
        )
        > merged.loc[away_favorite_mask, "set2_score1"]
    ).astype(int)
    return merged


def summarize_entries(df: pd.DataFrame, label: str) -> dict[str, float | int | str]:
    if df.empty:
        return {
            "group": label,
            "samples": 0,
            "matches": 0,
            "cover_rate": 0.0,
            "avg_odds": 0.0,
            "breakeven": 0.0,
            "edge": 0.0,
            "avg_entry_gap": 0.0,
            "avg_rally_number": 0.0,
        }

    cover_rate = float(df["favorite_set2_cover_target"].mean())
    breakeven = float((1.0 / df["favorite_set2_hcap_odds"]).mean())
    return {
        "group": label,
        "samples": int(len(df)),
        "matches": int(df["match_id"].nunique()),
        "cover_rate": cover_rate,
        "avg_odds": float(df["favorite_set2_hcap_odds"].mean()),
        "breakeven": breakeven,
        "edge": cover_rate - breakeven,
        "avg_entry_gap": float(df["favorite_score_gap_at_entry"].mean()),
        "avg_rally_number": float(df["rally_number"].mean()),
    }


def main() -> None:
    base_matches = build_base_matches()
    trajectory = build_trajectory_rows(base_matches)
    trajectory = attach_final_outcome(trajectory)

    opportunities = trajectory[trajectory["is_target_line"] == 1].copy()
    first_entry = (
        opportunities.sort_values(["match_id", "snapshot_ts", "rally_db_id"])
        .drop_duplicates(subset=["match_id"], keep="first")
        .copy()
    )
    best_odds_entry = (
        opportunities.sort_values(
            ["match_id", "favorite_set2_hcap_odds", "snapshot_ts", "rally_db_id"],
            ascending=[True, False, True, True],
        )
        .drop_duplicates(subset=["match_id"], keep="first")
        .copy()
    )
    dip_entry = opportunities[opportunities["favorite_score_gap_at_entry"] <= 0].copy()
    dip_first_entry = (
        dip_entry.sort_values(["match_id", "snapshot_ts", "rally_db_id"])
        .drop_duplicates(subset=["match_id"], keep="first")
        .copy()
    )

    matches_with_opportunity = opportunities["match_id"].nunique()
    total_matches = base_matches["match_id"].nunique()
    opportunity_rate = matches_with_opportunity / total_matches if total_matches else 0.0

    summary = pd.DataFrame(
        [
            summarize_entries(opportunities, "all_snapshots_at_-2.5"),
            summarize_entries(first_entry, "first_entry_at_-2.5"),
            summarize_entries(best_odds_entry, "best_odds_entry_at_-2.5"),
            summarize_entries(dip_first_entry, "first_entry_after_dip_at_-2.5"),
        ]
    )
    summary["target_hcap_line"] = TARGET_HCAP_LINE
    summary["base_matches"] = total_matches
    summary["matches_with_opportunity"] = matches_with_opportunity
    summary["opportunity_rate"] = opportunity_rate

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base_path = OUTPUT_DIR / "favorite_set1_hcap_reentry_base_matches.csv"
    trajectory_path = OUTPUT_DIR / "favorite_set1_hcap_reentry_opportunities.csv"
    summary_path = OUTPUT_DIR / "favorite_set1_hcap_reentry_summary.csv"

    base_matches.to_csv(base_path, index=False)
    opportunities.to_csv(trajectory_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("Favorite won set1 handicap re-entry analysis")
    print(
        "Scenario: opening if available else first_seen, "
        f"favorite_odds <= {TARGET_FAVORITE_ODDS_MAX:.2f}, "
        f"serve_gap >= {TARGET_SERVE_GAP_MIN:.2f}, favorite won set1."
    )
    print(f"Base matches: {total_matches}")
    print(f"Matches where favorite -2.5 appeared in set2: {matches_with_opportunity}")
    print(f"Opportunity rate: {opportunity_rate:.4f}")

    for row in summary.itertuples(index=False):
        print(
            f"  {row.group}: samples={row.samples}, "
            f"cover_rate={row.cover_rate:.4f}, "
            f"breakeven={row.breakeven:.4f}, "
            f"edge={row.edge:.4f}, "
            f"avg_odds={row.avg_odds:.2f}, "
            f"avg_entry_gap={row.avg_entry_gap:.2f}, "
            f"avg_rally={row.avg_rally_number:.2f}"
        )

    print(f"\nSummary saved to {summary_path}")
    print(f"Base matches saved to {base_path}")
    print(f"Opportunities saved to {trajectory_path}")


if __name__ == "__main__":
    main()
