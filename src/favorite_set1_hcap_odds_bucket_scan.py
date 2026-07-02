from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.favorite_set1_set2_side_markets_analysis import (
    build_analysis_rows,
    prepare_hcap_market,
)


OUTPUT_DIR = Path("data/processed")
TARGET_MAX_FAVORITE_ODDS = 1.50
TARGET_MIN_SERVE_GAP = 0.08
TARGET_MIN_SERVE_PCT = 0.55
TARGET_HCAP_LINE = -3.5
ODDS_BUCKETS = (
    (1.00, 1.15),
    (1.15, 1.20),
    (1.20, 1.25),
    (1.25, 1.30),
    (1.30, 1.35),
    (1.35, 1.40),
    (1.40, 1.45),
    (1.45, 1.50),
    (1.00, 1.20),
    (1.20, 1.35),
    (1.35, 1.50),
)
MIN_SAMPLES = 40


def select_pattern_rows() -> pd.DataFrame:
    rows = build_analysis_rows()
    hcap_rows = prepare_hcap_market(rows).copy()
    hcap_rows["match_date"] = pd.to_datetime(hcap_rows["match_date"])

    selected = hcap_rows[
        (hcap_rows["favorite_pre_match_odds"] <= TARGET_MAX_FAVORITE_ODDS)
        & (hcap_rows["favorite_set1_serve_gap"] >= TARGET_MIN_SERVE_GAP)
        & (hcap_rows["favorite_set1_serve_pct"] >= TARGET_MIN_SERVE_PCT)
        & (hcap_rows["favorite_set2_hcap_line"].round(2) == TARGET_HCAP_LINE)
    ].copy()

    selected = selected.sort_values(["match_date", "match_id"]).reset_index(drop=True)
    return selected


def summarize_group(df: pd.DataFrame, label: str) -> dict[str, float | int | str]:
    if df.empty:
        return {
            "group": label,
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
        "group": label,
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

    records.append(summarize_group(rows, "all_sources_all_odds"))
    for source in ("opening", "first_seen"):
        source_rows = rows[rows["odds_source"] == source].copy()
        records.append(summarize_group(source_rows, f"{source}_all_odds"))

        for min_odds, max_odds in ODDS_BUCKETS:
            scoped = source_rows[
                (source_rows["favorite_pre_match_odds"] >= min_odds)
                & (source_rows["favorite_pre_match_odds"] < max_odds)
            ].copy()
            records.append(
                summarize_group(
                    scoped,
                    f"{source}_{min_odds:.2f}_{max_odds:.2f}",
                )
            )

    summary = pd.DataFrame(records)
    summary = summary[summary["samples"] >= MIN_SAMPLES].copy()
    summary = summary.sort_values(
        ["edge", "samples"], ascending=[False, False]
    ).reset_index(drop=True)
    return summary


def main() -> None:
    rows = select_pattern_rows()
    summary = build_summary(rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows_path = OUTPUT_DIR / "favorite_set1_hcap_odds_bucket_rows.csv"
    summary_path = OUTPUT_DIR / "favorite_set1_hcap_odds_bucket_summary.csv"
    rows.to_csv(rows_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("Favorite won set1 handicap odds-bucket scan")
    print(
        "Pattern: opening if available else first_seen, "
        f"favorite_odds <= {TARGET_MAX_FAVORITE_ODDS:.2f}, "
        f"serve_gap >= {TARGET_MIN_SERVE_GAP:.2f}, "
        f"serve_pct >= {TARGET_MIN_SERVE_PCT:.2f}, "
        f"hcap_line = {TARGET_HCAP_LINE:.1f}"
    )
    print(f"Rows: {len(rows)}")
    print(f"Matches: {rows['match_id'].nunique() if not rows.empty else 0}")

    for row in summary.head(40).itertuples(index=False):
        print(
            f"  {row.group}: samples={row.samples}, "
            f"cover_rate={row.cover_rate:.4f}, "
            f"breakeven={row.breakeven:.4f}, "
            f"edge={row.edge:.4f}, "
            f"avg_hcap_odds={row.avg_hcap_odds:.2f}, "
            f"avg_favorite_odds={row.avg_favorite_odds:.2f}, "
            f"serve_gap={row.avg_serve_gap:.4f}, "
            f"serve_pct={row.avg_serve_pct:.4f}"
        )

    print(f"\nSummary saved to {summary_path}")
    print(f"Rows saved to {rows_path}")


if __name__ == "__main__":
    main()
