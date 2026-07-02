from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.db import load_favorite_set1_serve_rows
from src.favorite_set1_serve_analysis import prepare_rows
from src.live_set_db import load_current_set_live_rows


OUTPUT_DIR = Path("data/processed")
ODDS_RANGES = (
    (1.43, 1.50),
    (1.50, 1.60),
    (1.60, 1.70),
    (1.43, 1.60),
    (1.43, 1.70),
)


def load_first_set2_total_market_rows() -> pd.DataFrame:
    rows = load_current_set_live_rows(min_total_points=0, rally_step=1).copy()
    rows["snapshot_ts"] = pd.to_datetime(rows["snapshot_ts"])
    numeric_columns = [
        "set_number",
        "rally_number",
        "score1",
        "score2",
        "match_total_line",
        "match_total_over",
        "match_total_under",
    ]
    for column in numeric_columns:
        rows[column] = pd.to_numeric(rows[column], errors="coerce")

    rows = rows[
        (rows["set_number"] == 2)
        & rows["match_total_line"].notna()
        & rows["match_total_over"].notna()
        & rows["match_total_under"].notna()
    ].copy()

    rows = rows.sort_values(
        ["match_id", "snapshot_ts", "rally_number", "score1", "score2"]
    ).drop_duplicates(subset=["match_id"], keep="first")

    return rows[
        [
            "match_id",
            "snapshot_ts",
            "rally_number",
            "score1",
            "score2",
            "match_total_line",
            "match_total_over",
            "match_total_under",
        ]
    ].copy()


def build_analysis_rows() -> pd.DataFrame:
    historical = prepare_rows(load_favorite_set1_serve_rows()).copy()
    total_market = load_first_set2_total_market_rows()
    merged = historical.merge(total_market, on="match_id", how="inner")
    return merged


def enrich_with_total_result(rows: pd.DataFrame) -> pd.DataFrame:
    from sqlalchemy import create_engine, text

    from src.config import settings

    if not settings.db_url:
        raise ValueError("DB_URL is empty. Fill .env before loading total results.")

    query = """
WITH total_points AS (
    SELECT
        s.match_id,
        SUM(
            COALESCE(s.corrected_total, s.score1 + s.score2)
        ) AS total_match_points
    FROM sets s
    INNER JOIN matches m
        ON m.id = s.match_id
    WHERE COALESCE(s.finished, 0) = 1
      AND COALESCE(m.abandoned, 0) = 0
    GROUP BY s.match_id
)
SELECT
    match_id,
    total_match_points
FROM total_points
"""

    engine = create_engine(settings.db_url)
    with engine.connect() as connection:
        totals = pd.read_sql(text(query), connection)

    merged = rows.merge(totals, on="match_id", how="inner", suffixes=("", "_final"))
    merged["total_match_points"] = pd.to_numeric(
        merged["total_match_points_final"], errors="coerce"
    )
    merged = merged.drop(columns=["total_match_points_final"], errors="ignore")
    return merged


def prepare_market_outcomes(rows: pd.DataFrame) -> pd.DataFrame:
    df = rows.copy()
    df["over_hit"] = (df["total_match_points"] > df["match_total_line"]).astype(int)
    df["under_hit"] = (df["total_match_points"] < df["match_total_line"]).astype(int)
    df["push"] = (df["total_match_points"] == df["match_total_line"]).astype(int)

    df["market_side"] = "over"
    df.loc[df["match_total_under"] < df["match_total_over"], "market_side"] = "under"
    df["selected_odds"] = df["match_total_over"]
    df.loc[df["market_side"] == "under", "selected_odds"] = df.loc[
        df["market_side"] == "under", "match_total_under"
    ]

    df["selected_side_hit"] = df["over_hit"]
    df.loc[df["market_side"] == "under", "selected_side_hit"] = df.loc[
        df["market_side"] == "under", "under_hit"
    ]

    df["breakeven_rate"] = 1.0 / df["selected_odds"]
    df["edge_vs_breakeven"] = df["selected_side_hit"] - df["breakeven_rate"]
    return df[df["push"] == 0].copy()


def summarize_scope(
    rows: pd.DataFrame,
    label: str,
    side: str,
    min_odds: float,
    max_odds: float,
) -> dict[str, float | int | str]:
    scoped = rows[
        (rows["selected_odds"] >= min_odds)
        & (rows["selected_odds"] <= max_odds)
        & ((rows["market_side"] == side) if side != "all" else True)
    ].copy()

    if scoped.empty:
        return {
            "group": label,
            "market_side": side,
            "odds_range": f"{min_odds:.2f}-{max_odds:.2f}",
            "samples": 0,
            "matches": 0,
            "hit_rate": 0.0,
            "avg_odds": 0.0,
            "median_odds": 0.0,
            "avg_line": 0.0,
            "avg_breakeven_rate": 0.0,
            "edge_vs_breakeven": 0.0,
            "avg_rally_number": 0.0,
            "avg_score_total": 0.0,
        }

    return {
        "group": label,
        "market_side": side,
        "odds_range": f"{min_odds:.2f}-{max_odds:.2f}",
        "samples": int(len(scoped)),
        "matches": int(scoped["match_id"].nunique()),
        "hit_rate": float(scoped["selected_side_hit"].mean()),
        "avg_odds": float(scoped["selected_odds"].mean()),
        "median_odds": float(scoped["selected_odds"].median()),
        "avg_line": float(scoped["match_total_line"].mean()),
        "avg_breakeven_rate": float(scoped["breakeven_rate"].mean()),
        "edge_vs_breakeven": float(
            scoped["selected_side_hit"].mean() - scoped["breakeven_rate"].mean()
        ),
        "avg_rally_number": float(scoped["rally_number"].mean()),
        "avg_score_total": float((scoped["score1"] + scoped["score2"]).mean()),
    }


def build_summary(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    for min_odds, max_odds in ODDS_RANGES:
        for side in ("all", "over", "under"):
            records.append(
                summarize_scope(
                    rows,
                    "post_set1_match_total",
                    side,
                    min_odds,
                    max_odds,
                )
            )
    return pd.DataFrame(records).sort_values(
        ["odds_range", "market_side"]
    ).reset_index(drop=True)


def main() -> None:
    rows = build_analysis_rows()
    rows = enrich_with_total_result(rows)
    rows = prepare_market_outcomes(rows)
    summary = build_summary(rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows_path = OUTPUT_DIR / "post_set1_match_total_rows.csv"
    summary_path = OUTPUT_DIR / "post_set1_match_total_summary.csv"
    rows.to_csv(rows_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("Post-set1 match total analysis")
    print("Scenario: first available match-total market in set 2.")
    print(f"Rows (excluding pushes): {len(rows)}")
    print(f"Matches: {rows['match_id'].nunique()}")

    for odds_range in summary["odds_range"].drop_duplicates().tolist():
        print(f"\nOdds range {odds_range}:")
        scoped = summary[summary["odds_range"] == odds_range]
        for row in scoped.itertuples(index=False):
            print(
                f"  {row.market_side}: samples={row.samples}, "
                f"hit_rate={row.hit_rate:.4f}, "
                f"breakeven={row.avg_breakeven_rate:.4f}, "
                f"edge={row.edge_vs_breakeven:.4f}, "
                f"avg_odds={row.avg_odds:.2f}, "
                f"avg_line={row.avg_line:.2f}, "
                f"avg_rally={row.avg_rally_number:.2f}, "
                f"avg_score_total={row.avg_score_total:.2f}"
            )

    print(f"\nSummary saved to {summary_path}")
    print(f"Prepared rows saved to {rows_path}")


if __name__ == "__main__":
    main()
