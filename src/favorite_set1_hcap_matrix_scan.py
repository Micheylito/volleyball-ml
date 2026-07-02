from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.favorite_set1_set2_side_markets_analysis import (
    build_analysis_rows,
    prepare_hcap_market,
)


OUTPUT_DIR = Path("data/processed")
TARGET_HCAP_LINE = -3.5
MIN_SAMPLES = 30

ODDS_BUCKETS = (
    (1.00, 1.15),
    (1.15, 1.20),
    (1.20, 1.35),
    (1.35, 1.50),
    (1.50, 1.65),
    (1.65, 1.80),
    (1.00, 1.20),
    (1.20, 1.35),
    (1.35, 1.50),
    (1.50, 1.80),
    (1.00, 1.50),
    (1.00, 1.80),
)

SERVE_PCT_THRESHOLDS = (0.50, 0.55, 0.60)
SERVE_GAP_THRESHOLDS = (0.03, 0.05, 0.08, 0.10)


def load_pattern_rows() -> pd.DataFrame:
    rows = build_analysis_rows()
    hcap_rows = prepare_hcap_market(rows).copy()
    hcap_rows["match_date"] = pd.to_datetime(hcap_rows["match_date"])

    selected = hcap_rows[
        hcap_rows["favorite_set2_hcap_line"].round(2) == TARGET_HCAP_LINE
    ].copy()
    selected = selected.sort_values(["match_date", "match_id"]).reset_index(drop=True)
    return selected


def summarize_group(
    df: pd.DataFrame,
    *,
    source: str,
    odds_bucket: str,
    serve_pct_threshold: float,
    serve_gap_threshold: float,
) -> dict[str, float | int | str]:
    if df.empty:
        return {
            "source": source,
            "odds_bucket": odds_bucket,
            "serve_pct_threshold": serve_pct_threshold,
            "serve_gap_threshold": serve_gap_threshold,
            "samples": 0,
            "matches": 0,
            "cover_rate": 0.0,
            "breakeven": 0.0,
            "edge": 0.0,
            "avg_hcap_odds": 0.0,
            "avg_favorite_odds": 0.0,
            "avg_serve_gap": 0.0,
            "avg_serve_pct": 0.0,
            "date_from": "",
            "date_to": "",
        }

    breakeven = float((1.0 / df["favorite_set2_hcap_odds"]).mean())
    cover_rate = float(df["favorite_set2_cover"].mean())
    return {
        "source": source,
        "odds_bucket": odds_bucket,
        "serve_pct_threshold": serve_pct_threshold,
        "serve_gap_threshold": serve_gap_threshold,
        "samples": int(len(df)),
        "matches": int(df["match_id"].nunique()),
        "cover_rate": cover_rate,
        "breakeven": breakeven,
        "edge": cover_rate - breakeven,
        "avg_hcap_odds": float(df["favorite_set2_hcap_odds"].mean()),
        "avg_favorite_odds": float(df["favorite_pre_match_odds"].mean()),
        "avg_serve_gap": float(df["favorite_set1_serve_gap"].mean()),
        "avg_serve_pct": float(df["favorite_set1_serve_pct"].mean()),
        "date_from": str(df["match_date"].min()),
        "date_to": str(df["match_date"].max()),
    }


def build_summary(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []

    for source in ("opening", "first_seen"):
        source_rows = rows[rows["odds_source"] == source].copy()
        for min_odds, max_odds in ODDS_BUCKETS:
            odds_rows = source_rows[
                (source_rows["favorite_pre_match_odds"] >= min_odds)
                & (source_rows["favorite_pre_match_odds"] < max_odds)
            ].copy()
            odds_bucket = f"{min_odds:.2f}-{max_odds:.2f}"

            for serve_pct_threshold in SERVE_PCT_THRESHOLDS:
                pct_rows = odds_rows[
                    odds_rows["favorite_set1_serve_pct"] >= serve_pct_threshold
                ].copy()

                for serve_gap_threshold in SERVE_GAP_THRESHOLDS:
                    scoped = pct_rows[
                        pct_rows["favorite_set1_serve_gap"] >= serve_gap_threshold
                    ].copy()
                    records.append(
                        summarize_group(
                            scoped,
                            source=source,
                            odds_bucket=odds_bucket,
                            serve_pct_threshold=serve_pct_threshold,
                            serve_gap_threshold=serve_gap_threshold,
                        )
                    )

    summary = pd.DataFrame(records)
    summary = summary[summary["samples"] >= MIN_SAMPLES].copy()
    summary = summary.sort_values(
        ["source", "edge", "samples"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    return summary


def print_source_summary(summary: pd.DataFrame, source: str, limit: int = 20) -> None:
    scoped = summary[summary["source"] == source].copy()
    if scoped.empty:
        print(f"\n{source}: нет групп с samples >= {MIN_SAMPLES}")
        return

    print(f"\nTop {source} groups:")
    for row in scoped.head(limit).itertuples(index=False):
        print(
            f"  odds={row.odds_bucket}, "
            f"serve_pct>={row.serve_pct_threshold:.2f}, "
            f"serve_gap>={row.serve_gap_threshold:.2f}: "
            f"samples={row.samples}, "
            f"cover_rate={row.cover_rate:.4f}, "
            f"breakeven={row.breakeven:.4f}, "
            f"edge={row.edge:.4f}, "
            f"avg_hcap_odds={row.avg_hcap_odds:.2f}, "
            f"avg_favorite_odds={row.avg_favorite_odds:.2f}, "
            f"avg_serve_gap={row.avg_serve_gap:.4f}, "
            f"avg_serve_pct={row.avg_serve_pct:.4f}"
        )


def main() -> None:
    rows = load_pattern_rows()
    summary = build_summary(rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows_path = OUTPUT_DIR / "favorite_set1_hcap_matrix_rows.csv"
    summary_path = OUTPUT_DIR / "favorite_set1_hcap_matrix_summary.csv"

    rows.to_csv(rows_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("Favorite won set1 handicap matrix scan")
    print(
        "Scenario: opening/first_seen separately, "
        "scan odds buckets with serve_pct and serve_gap thresholds, "
        f"hcap_line = {TARGET_HCAP_LINE:.1f}"
    )
    print(f"Rows: {len(rows)}")
    print(f"Matches: {rows['match_id'].nunique() if not rows.empty else 0}")
    print(f"Min samples filter: {MIN_SAMPLES}")

    print_source_summary(summary, "opening")
    print_source_summary(summary, "first_seen")

    print(f"\nSummary saved to {summary_path}")
    print(f"Rows saved to {rows_path}")


if __name__ == "__main__":
    main()
