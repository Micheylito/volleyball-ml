from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.favorite_set1_serve_analysis import prepare_rows
from src.db import load_favorite_set1_serve_rows
from src.live_set_db import load_first_set2_match_total_rows
from src.post_set1_match_total_analysis import enrich_with_total_result


OUTPUT_DIR = Path("data/processed")
FAVORITE_ODDS_THRESHOLDS = (1.50, 1.60, 1.70)
SERVE_GAP_THRESHOLDS = (0.05, 0.08)


def build_analysis_rows() -> pd.DataFrame:
    historical = prepare_rows(load_favorite_set1_serve_rows()).copy()
    historical = historical[historical["favorite_won_set1"] == 1].copy()

    historical["favorite_pre_match_odds"] = historical["home_odds"]
    away_favorite_mask = historical["favorite_team"] == 2
    historical.loc[away_favorite_mask, "favorite_pre_match_odds"] = historical.loc[
        away_favorite_mask, "away_odds"
    ]

    total_market = load_first_set2_match_total_rows().copy()
    total_market["snapshot_ts"] = pd.to_datetime(total_market["snapshot_ts"])
    for column in [
        "rally_number",
        "score1",
        "score2",
        "match_total_line",
        "match_total_over",
        "match_total_under",
    ]:
        total_market[column] = pd.to_numeric(total_market[column], errors="coerce")

    merged = historical.merge(total_market, on="match_id", how="inner")
    merged = enrich_with_total_result(merged)

    merged["over_hit"] = (merged["total_match_points"] > merged["match_total_line"]).astype(int)
    merged["under_hit"] = (merged["total_match_points"] < merged["match_total_line"]).astype(int)
    merged["push"] = (merged["total_match_points"] == merged["match_total_line"]).astype(int)
    merged = merged[merged["push"] == 0].copy()

    merged["favorite_pre_match_odds"] = pd.to_numeric(
        merged["favorite_pre_match_odds"], errors="coerce"
    )
    merged["favorite_set1_serve_gap"] = pd.to_numeric(
        merged["favorite_set1_serve_gap"], errors="coerce"
    )
    merged["favorite_set1_serve_pct"] = pd.to_numeric(
        merged["favorite_set1_serve_pct"], errors="coerce"
    )
    merged["match_total_over"] = pd.to_numeric(merged["match_total_over"], errors="coerce")
    merged["match_total_under"] = pd.to_numeric(merged["match_total_under"], errors="coerce")
    merged["match_total_line"] = pd.to_numeric(merged["match_total_line"], errors="coerce")
    return merged


def summarize_group(df: pd.DataFrame, label: str) -> dict[str, float | int | str]:
    if df.empty:
        return {
            "group": label,
            "samples": 0,
            "matches": 0,
            "avg_favorite_pre_match_odds": 0.0,
            "avg_serve_gap": 0.0,
            "avg_serve_pct": 0.0,
            "avg_total_line": 0.0,
            "avg_snapshot_score_total": 0.0,
            "avg_rally_number": 0.0,
            "over_hit_rate": 0.0,
            "under_hit_rate": 0.0,
            "avg_over_odds": 0.0,
            "avg_under_odds": 0.0,
            "over_breakeven": 0.0,
            "under_breakeven": 0.0,
            "over_edge": 0.0,
            "under_edge": 0.0,
            "better_side": "none",
        }

    over_breakeven = float((1.0 / df["match_total_over"]).mean())
    under_breakeven = float((1.0 / df["match_total_under"]).mean())
    over_hit_rate = float(df["over_hit"].mean())
    under_hit_rate = float(df["under_hit"].mean())
    over_edge = over_hit_rate - over_breakeven
    under_edge = under_hit_rate - under_breakeven
    if over_edge > under_edge:
        better_side = "over"
    elif under_edge > over_edge:
        better_side = "under"
    else:
        better_side = "equal"

    return {
        "group": label,
        "samples": int(len(df)),
        "matches": int(df["match_id"].nunique()),
        "avg_favorite_pre_match_odds": float(df["favorite_pre_match_odds"].mean()),
        "avg_serve_gap": float(df["favorite_set1_serve_gap"].mean()),
        "avg_serve_pct": float(df["favorite_set1_serve_pct"].mean()),
        "avg_total_line": float(df["match_total_line"].mean()),
        "avg_snapshot_score_total": float((df["score1"] + df["score2"]).mean()),
        "avg_rally_number": float(df["rally_number"].mean()),
        "over_hit_rate": over_hit_rate,
        "under_hit_rate": under_hit_rate,
        "avg_over_odds": float(df["match_total_over"].mean()),
        "avg_under_odds": float(df["match_total_under"].mean()),
        "over_breakeven": over_breakeven,
        "under_breakeven": under_breakeven,
        "over_edge": over_edge,
        "under_edge": under_edge,
        "better_side": better_side,
    }


def build_summary(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []

    for source in ("opening", "first_seen"):
        source_rows = rows[rows["odds_source"] == source].copy()
        records.append(summarize_group(source_rows, f"{source}_all"))

        for fav_threshold in FAVORITE_ODDS_THRESHOLDS:
            fav_rows = source_rows[source_rows["favorite_pre_match_odds"] <= fav_threshold].copy()
            records.append(
                summarize_group(fav_rows, f"{source}_favorite_odds_le_{fav_threshold:.2f}")
            )

            for serve_gap_threshold in SERVE_GAP_THRESHOLDS:
                scoped = fav_rows[
                    fav_rows["favorite_set1_serve_gap"] >= serve_gap_threshold
                ].copy()
                records.append(
                    summarize_group(
                        scoped,
                        f"{source}_favorite_odds_le_{fav_threshold:.2f}"
                        f"_serve_gap_ge_{serve_gap_threshold:.2f}",
                    )
                )

    return pd.DataFrame(records)


def main() -> None:
    rows = build_analysis_rows()
    summary = build_summary(rows)
    summary = summary.sort_values(
        ["better_side", "under_edge", "over_edge", "samples"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows_path = OUTPUT_DIR / "favorite_set1_total_direction_rows.csv"
    summary_path = OUTPUT_DIR / "favorite_set1_total_direction_summary.csv"
    rows.to_csv(rows_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("Favorite won set1 total-direction analysis")
    print("Scenario: strong pre-match favorite + good serve in set 1 + inspect match total after set 1.")
    print(f"Rows (excluding pushes): {len(rows)}")
    print(f"Matches: {rows['match_id'].nunique()}")

    for row in summary.itertuples(index=False):
        if row.samples == 0:
            continue
        print(
            f"  {row.group}: samples={row.samples}, "
            f"fav_odds={row.avg_favorite_pre_match_odds:.2f}, "
            f"serve_gap={row.avg_serve_gap:.4f}, "
            f"line={row.avg_total_line:.2f}, "
            f"over_hit={row.over_hit_rate:.4f}, over_edge={row.over_edge:.4f}, "
            f"under_hit={row.under_hit_rate:.4f}, under_edge={row.under_edge:.4f}, "
            f"better_side={row.better_side}"
        )

    print(f"Summary saved to {summary_path}")
    print(f"Prepared rows saved to {rows_path}")


if __name__ == "__main__":
    main()
