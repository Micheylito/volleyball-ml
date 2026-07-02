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
SERVE_GAP_THRESHOLDS = (0.00, 0.05, 0.08)


def load_first_set2_market_rows() -> pd.DataFrame:
    rows = load_current_set_live_rows(min_total_points=0, rally_step=1).copy()
    rows["snapshot_ts"] = pd.to_datetime(rows["snapshot_ts"])
    rows["set_number"] = pd.to_numeric(rows["set_number"], errors="coerce")
    rows["rally_number"] = pd.to_numeric(rows["rally_number"], errors="coerce")
    rows["score1"] = pd.to_numeric(rows["score1"], errors="coerce")
    rows["score2"] = pd.to_numeric(rows["score2"], errors="coerce")
    rows["set_win1"] = pd.to_numeric(rows["set_win1"], errors="coerce")
    rows["set_win2"] = pd.to_numeric(rows["set_win2"], errors="coerce")

    rows = rows[
        (rows["set_number"] == 2)
        & rows["set_win1"].notna()
        & rows["set_win2"].notna()
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
            "set_win1",
            "set_win2",
        ]
    ].copy()


def build_analysis_rows() -> pd.DataFrame:
    historical = prepare_rows(load_favorite_set1_serve_rows())
    historical = historical[historical["favorite_won_set1"] == 1].copy()

    set2_market = load_first_set2_market_rows()
    merged = historical.merge(set2_market, on="match_id", how="inner")

    merged["favorite_set2_odds"] = merged["set_win1"]
    merged["opponent_set2_odds"] = merged["set_win2"]
    away_favorite_mask = merged["favorite_team"] == 2
    merged.loc[away_favorite_mask, "favorite_set2_odds"] = merged.loc[
        away_favorite_mask, "set_win2"
    ]
    merged.loc[away_favorite_mask, "opponent_set2_odds"] = merged.loc[
        away_favorite_mask, "set_win1"
    ]

    merged["favorite_is_set2_market_favorite"] = (
        merged["favorite_set2_odds"] < merged["opponent_set2_odds"]
    ).astype(int)
    merged["set2_market_flip_against_favorite"] = (
        merged["favorite_is_set2_market_favorite"] == 0
    ).astype(int)

    merged["favorite_set2_odds"] = pd.to_numeric(
        merged["favorite_set2_odds"], errors="coerce"
    )
    merged["favorite_won_set2"] = pd.to_numeric(
        merged["favorite_won_set2"], errors="coerce"
    )
    merged["favorite_set1_serve_gap"] = pd.to_numeric(
        merged["favorite_set1_serve_gap"], errors="coerce"
    )
    merged["favorite_set1_serve_pct"] = pd.to_numeric(
        merged["favorite_set1_serve_pct"], errors="coerce"
    )
    return merged


def summarize_scope(
    rows: pd.DataFrame,
    label: str,
    min_odds: float,
    max_odds: float,
    min_serve_gap: float,
) -> dict[str, float | int | str]:
    scoped = rows[
        (rows["favorite_set2_odds"] >= min_odds)
        & (rows["favorite_set2_odds"] <= max_odds)
        & (rows["favorite_set1_serve_gap"] >= min_serve_gap)
    ].copy()

    if scoped.empty:
        return {
            "group": label,
            "odds_range": f"{min_odds:.2f}-{max_odds:.2f}",
            "min_serve_gap": min_serve_gap,
            "samples": 0,
            "matches": 0,
            "win_set2_rate": 0.0,
            "avg_set2_odds": 0.0,
            "median_set2_odds": 0.0,
            "avg_serve_gap": 0.0,
            "avg_serve_pct": 0.0,
            "set2_market_favorite_rate": 0.0,
            "set2_market_flip_rate": 0.0,
            "avg_rally_number": 0.0,
            "avg_snapshot_score_total": 0.0,
        }

    return {
        "group": label,
        "odds_range": f"{min_odds:.2f}-{max_odds:.2f}",
        "min_serve_gap": min_serve_gap,
        "samples": int(len(scoped)),
        "matches": int(scoped["match_id"].nunique()),
        "win_set2_rate": float(scoped["favorite_won_set2"].mean()),
        "avg_set2_odds": float(scoped["favorite_set2_odds"].mean()),
        "median_set2_odds": float(scoped["favorite_set2_odds"].median()),
        "avg_serve_gap": float(scoped["favorite_set1_serve_gap"].mean()),
        "avg_serve_pct": float(scoped["favorite_set1_serve_pct"].mean()),
        "set2_market_favorite_rate": float(
            scoped["favorite_is_set2_market_favorite"].mean()
        ),
        "set2_market_flip_rate": float(
            scoped["set2_market_flip_against_favorite"].mean()
        ),
        "avg_rally_number": float(scoped["rally_number"].mean()),
        "avg_snapshot_score_total": float((scoped["score1"] + scoped["score2"]).mean()),
    }


def build_summary(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    for min_odds, max_odds in ODDS_RANGES:
        baseline = summarize_scope(rows, "baseline", min_odds, max_odds, 0.0)
        records.append(baseline)
        for threshold in SERVE_GAP_THRESHOLDS[1:]:
            records.append(
                summarize_scope(
                    rows,
                    f"serve_gap_ge_{threshold:.2f}",
                    min_odds,
                    max_odds,
                    threshold,
                )
            )

    summary = pd.DataFrame(records)
    baseline_rates = (
        summary[summary["group"] == "baseline"][["odds_range", "win_set2_rate"]]
        .rename(columns={"win_set2_rate": "baseline_win_set2_rate"})
        .copy()
    )
    summary = summary.merge(baseline_rates, on="odds_range", how="left")
    summary["uplift_vs_range_baseline"] = (
        summary["win_set2_rate"] - summary["baseline_win_set2_rate"]
    )
    return summary.sort_values(["odds_range", "min_serve_gap"]).reset_index(drop=True)


def main() -> None:
    rows = build_analysis_rows()
    summary = build_summary(rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows_path = OUTPUT_DIR / "favorite_set1_set2_market_rows.csv"
    summary_path = OUTPUT_DIR / "favorite_set1_set2_market_summary.csv"
    rows.to_csv(rows_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("Favorite set1 + set2 market analysis")
    print("Scenario: favorite won set1, then we inspect the first available set2 market.")
    print(f"Rows: {len(rows)}")
    print(f"Matches: {rows['match_id'].nunique()}")

    for odds_range in summary["odds_range"].drop_duplicates().tolist():
        print(f"\nSet2 odds range {odds_range}:")
        scoped = summary[summary["odds_range"] == odds_range]
        for row in scoped.itertuples(index=False):
            print(
                f"  {row.group}: samples={row.samples}, "
                f"win_set2_rate={row.win_set2_rate:.4f}, "
                f"uplift={row.uplift_vs_range_baseline:.4f}, "
                f"avg_odds={row.avg_set2_odds:.2f}, "
                f"serve_gap={row.avg_serve_gap:.4f}, "
                f"market_fav_rate={row.set2_market_favorite_rate:.4f}, "
                f"flip_rate={row.set2_market_flip_rate:.4f}, "
                f"avg_rally={row.avg_rally_number:.2f}, "
                f"avg_score_total={row.avg_snapshot_score_total:.2f}"
            )

    print(f"\nSummary saved to {summary_path}")
    print(f"Prepared rows saved to {rows_path}")


if __name__ == "__main__":
    main()
