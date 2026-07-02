from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from src.config import settings
from src.db import load_favorite_set1_serve_rows
from src.favorite_set1_serve_analysis import prepare_rows
from src.live_set_db import load_first_set2_side_market_rows


OUTPUT_DIR = Path("data/processed")
FAVORITE_ODDS_THRESHOLDS = (1.50, 1.60, 1.70)
SERVE_GAP_THRESHOLDS = (0.05, 0.08)


SET2_FINAL_SCORES_QUERY = """
SELECT
    s.match_id,
    s.score1 AS set2_score1,
    s.score2 AS set2_score2,
    COALESCE(s.corrected_total, s.score1 + s.score2) AS set2_total_points
FROM sets s
INNER JOIN matches m
    ON m.id = s.match_id
WHERE s.set_number = 2
  AND COALESCE(s.finished, 0) = 1
  AND COALESCE(m.abandoned, 0) = 0
"""


def load_set2_final_scores() -> pd.DataFrame:
    if not settings.db_url:
        raise ValueError("DB_URL is empty. Fill .env before loading set2 final scores.")

    engine = create_engine(settings.db_url)
    with engine.connect() as connection:
        rows = pd.read_sql(text(SET2_FINAL_SCORES_QUERY), connection)

    if rows.empty:
        raise ValueError("The set2 final score query returned no rows.")

    return rows


def build_analysis_rows() -> pd.DataFrame:
    historical = prepare_rows(load_favorite_set1_serve_rows()).copy()
    historical = historical[historical["favorite_won_set1"] == 1].copy()

    historical["favorite_pre_match_odds"] = historical["home_odds"]
    away_favorite_mask = historical["favorite_team"] == 2
    historical.loc[away_favorite_mask, "favorite_pre_match_odds"] = historical.loc[
        away_favorite_mask, "away_odds"
    ]

    market_rows = load_first_set2_side_market_rows().copy()
    market_rows["snapshot_ts"] = pd.to_datetime(market_rows["snapshot_ts"])
    for column in [
        "rally_number",
        "score1",
        "score2",
        "set_total_line",
        "set_total_over",
        "set_total_under",
        "set_hcap_line",
        "set_hcap1",
        "set_hcap2",
    ]:
        market_rows[column] = pd.to_numeric(market_rows[column], errors="coerce")

    set2_scores = load_set2_final_scores()
    for column in ["set2_score1", "set2_score2", "set2_total_points"]:
        set2_scores[column] = pd.to_numeric(set2_scores[column], errors="coerce")

    merged = historical.merge(market_rows, on="match_id", how="inner")
    merged = merged.merge(set2_scores, on="match_id", how="inner")

    merged["favorite_pre_match_odds"] = pd.to_numeric(
        merged["favorite_pre_match_odds"], errors="coerce"
    )
    merged["favorite_set1_serve_gap"] = pd.to_numeric(
        merged["favorite_set1_serve_gap"], errors="coerce"
    )
    merged["favorite_set1_serve_pct"] = pd.to_numeric(
        merged["favorite_set1_serve_pct"], errors="coerce"
    )
    return merged


def prepare_total_market(rows: pd.DataFrame) -> pd.DataFrame:
    df = rows.copy()
    df = df[
        df["set_total_line"].notna()
        & df["set_total_over"].notna()
        & df["set_total_under"].notna()
    ].copy()

    df["over_hit"] = (df["set2_total_points"] > df["set_total_line"]).astype(int)
    df["under_hit"] = (df["set2_total_points"] < df["set_total_line"]).astype(int)
    df["push"] = (df["set2_total_points"] == df["set_total_line"]).astype(int)
    df = df[df["push"] == 0].copy()
    return df


def prepare_hcap_market(rows: pd.DataFrame) -> pd.DataFrame:
    df = rows.copy()
    df = df[
        df["set_hcap_line"].notna()
        & df["set_hcap1"].notna()
        & df["set_hcap2"].notna()
    ].copy()

    df["favorite_set2_hcap_odds"] = df["set_hcap1"]
    df["favorite_set2_cover"] = (
        (df["set2_score1"] + df["set_hcap_line"]) > df["set2_score2"]
    ).astype(int)

    away_favorite_mask = df["favorite_team"] == 2
    df.loc[away_favorite_mask, "favorite_set2_hcap_odds"] = df.loc[
        away_favorite_mask, "set_hcap2"
    ]
    df.loc[away_favorite_mask, "favorite_set2_cover"] = (
        (
            df.loc[away_favorite_mask, "set2_score2"]
            + df.loc[away_favorite_mask, "set_hcap_line"]
        )
        > df.loc[away_favorite_mask, "set2_score1"]
    ).astype(int)

    return df


def summarize_total_group(df: pd.DataFrame, label: str) -> dict[str, float | int | str]:
    if df.empty:
        return {
            "market": "set2_total",
            "group": label,
            "samples": 0,
            "avg_favorite_pre_match_odds": 0.0,
            "avg_serve_gap": 0.0,
            "avg_line": 0.0,
            "over_hit_rate": 0.0,
            "under_hit_rate": 0.0,
            "avg_over_odds": 0.0,
            "avg_under_odds": 0.0,
            "over_edge": 0.0,
            "under_edge": 0.0,
            "better_side": "none",
        }

    over_breakeven = float((1.0 / df["set_total_over"]).mean())
    under_breakeven = float((1.0 / df["set_total_under"]).mean())
    over_hit_rate = float(df["over_hit"].mean())
    under_hit_rate = float(df["under_hit"].mean())
    over_edge = over_hit_rate - over_breakeven
    under_edge = under_hit_rate - under_breakeven
    better_side = "over" if over_edge > under_edge else "under"

    return {
        "market": "set2_total",
        "group": label,
        "samples": int(len(df)),
        "avg_favorite_pre_match_odds": float(df["favorite_pre_match_odds"].mean()),
        "avg_serve_gap": float(df["favorite_set1_serve_gap"].mean()),
        "avg_line": float(df["set_total_line"].mean()),
        "over_hit_rate": over_hit_rate,
        "under_hit_rate": under_hit_rate,
        "avg_over_odds": float(df["set_total_over"].mean()),
        "avg_under_odds": float(df["set_total_under"].mean()),
        "over_edge": over_edge,
        "under_edge": under_edge,
        "better_side": better_side,
    }


def summarize_hcap_group(df: pd.DataFrame, label: str) -> dict[str, float | int | str]:
    if df.empty:
        return {
            "market": "set2_hcap",
            "group": label,
            "samples": 0,
            "avg_favorite_pre_match_odds": 0.0,
            "avg_serve_gap": 0.0,
            "avg_line": 0.0,
            "favorite_cover_rate": 0.0,
            "avg_favorite_hcap_odds": 0.0,
            "favorite_cover_edge": 0.0,
            "better_side": "none",
        }

    cover_rate = float(df["favorite_set2_cover"].mean())
    breakeven = float((1.0 / df["favorite_set2_hcap_odds"]).mean())
    cover_edge = cover_rate - breakeven

    return {
        "market": "set2_hcap",
        "group": label,
        "samples": int(len(df)),
        "avg_favorite_pre_match_odds": float(df["favorite_pre_match_odds"].mean()),
        "avg_serve_gap": float(df["favorite_set1_serve_gap"].mean()),
        "avg_line": float(df["set_hcap_line"].mean()),
        "favorite_cover_rate": cover_rate,
        "avg_favorite_hcap_odds": float(df["favorite_set2_hcap_odds"].mean()),
        "favorite_cover_edge": cover_edge,
        "better_side": "favorite_hcap",
    }


def build_group_specs(rows: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    groups: list[tuple[str, pd.DataFrame]] = []
    for source in ("opening", "first_seen"):
        source_rows = rows[rows["odds_source"] == source].copy()
        groups.append((f"{source}_all", source_rows))

        for fav_threshold in FAVORITE_ODDS_THRESHOLDS:
            fav_rows = source_rows[source_rows["favorite_pre_match_odds"] <= fav_threshold].copy()
            groups.append((f"{source}_favorite_odds_le_{fav_threshold:.2f}", fav_rows))

            for serve_gap_threshold in SERVE_GAP_THRESHOLDS:
                scoped = fav_rows[
                    fav_rows["favorite_set1_serve_gap"] >= serve_gap_threshold
                ].copy()
                groups.append(
                    (
                        f"{source}_favorite_odds_le_{fav_threshold:.2f}"
                        f"_serve_gap_ge_{serve_gap_threshold:.2f}",
                        scoped,
                    )
                )
    return groups


def main() -> None:
    rows = build_analysis_rows()
    total_rows = prepare_total_market(rows)
    hcap_rows = prepare_hcap_market(rows)

    records: list[dict[str, float | int | str]] = []
    for label, scoped_rows in build_group_specs(rows):
        total_scoped = total_rows[total_rows["match_id"].isin(scoped_rows["match_id"])].copy()
        hcap_scoped = hcap_rows[hcap_rows["match_id"].isin(scoped_rows["match_id"])].copy()
        records.append(summarize_total_group(total_scoped, label))
        records.append(summarize_hcap_group(hcap_scoped, label))

    summary = pd.DataFrame(records)
    total_summary = summary[summary["market"] == "set2_total"].copy()
    hcap_summary = summary[summary["market"] == "set2_hcap"].copy()

    total_summary = total_summary.sort_values(
        ["under_edge", "over_edge", "samples"], ascending=[False, False, False]
    ).reset_index(drop=True)
    hcap_summary = hcap_summary.sort_values(
        ["favorite_cover_edge", "samples"], ascending=[False, False]
    ).reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_rows_path = OUTPUT_DIR / "favorite_set1_set2_total_rows.csv"
    total_summary_path = OUTPUT_DIR / "favorite_set1_set2_total_summary.csv"
    hcap_rows_path = OUTPUT_DIR / "favorite_set1_set2_hcap_rows.csv"
    hcap_summary_path = OUTPUT_DIR / "favorite_set1_set2_hcap_summary.csv"

    total_rows.to_csv(total_rows_path, index=False)
    total_summary.to_csv(total_summary_path, index=False)
    hcap_rows.to_csv(hcap_rows_path, index=False)
    hcap_summary.to_csv(hcap_summary_path, index=False)

    print("Favorite won set1 side-markets analysis")
    print("Scenario: strong pre-match favorite + good serve in set 1, then inspect set2 total and handicap.")
    print(f"Base matches: {rows['match_id'].nunique()}")
    print(f"Set2 total rows: {len(total_rows)}")
    print(f"Set2 handicap rows: {len(hcap_rows)}")

    print("\nTop set2 total groups:")
    for row in total_summary.head(20).itertuples(index=False):
        if row.samples == 0:
            continue
        print(
            f"  {row.group}: samples={row.samples}, "
            f"fav_odds={row.avg_favorite_pre_match_odds:.2f}, "
            f"serve_gap={row.avg_serve_gap:.4f}, "
            f"line={row.avg_line:.2f}, "
            f"over_edge={row.over_edge:.4f}, "
            f"under_edge={row.under_edge:.4f}, "
            f"better_side={row.better_side}"
        )

    print("\nTop set2 handicap groups:")
    for row in hcap_summary.head(20).itertuples(index=False):
        if row.samples == 0:
            continue
        print(
            f"  {row.group}: samples={row.samples}, "
            f"fav_odds={row.avg_favorite_pre_match_odds:.2f}, "
            f"serve_gap={row.avg_serve_gap:.4f}, "
            f"hcap_line={row.avg_line:.2f}, "
            f"cover_rate={row.favorite_cover_rate:.4f}, "
            f"cover_edge={row.favorite_cover_edge:.4f}"
        )

    print(f"\nSet2 total summary saved to {total_summary_path}")
    print(f"Set2 total rows saved to {total_rows_path}")
    print(f"Set2 handicap summary saved to {hcap_summary_path}")
    print(f"Set2 handicap rows saved to {hcap_rows_path}")


if __name__ == "__main__":
    main()
