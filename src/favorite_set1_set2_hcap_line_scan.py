from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.favorite_set1_set2_side_markets_analysis import (
    build_analysis_rows,
    prepare_hcap_market,
)


OUTPUT_DIR = Path("data/processed")
MIN_SAMPLES = 150


def normalize_hcap_line(value: float) -> str:
    rounded = round(float(value) * 2) / 2
    if rounded.is_integer():
        return f"{int(rounded)}.0"
    return f"{rounded:.1f}"


def summarize_group(df: pd.DataFrame, label: str) -> dict[str, float | int | str]:
    if df.empty:
        return {
            "group": label,
            "samples": 0,
            "matches": 0,
            "avg_favorite_pre_match_odds": 0.0,
            "avg_serve_gap": 0.0,
            "avg_hcap_line": 0.0,
            "avg_favorite_hcap_odds": 0.0,
            "cover_rate": 0.0,
            "breakeven_rate": 0.0,
            "cover_edge": 0.0,
        }

    breakeven = float((1.0 / df["favorite_set2_hcap_odds"]).mean())
    cover_rate = float(df["favorite_set2_cover"].mean())
    return {
        "group": label,
        "samples": int(len(df)),
        "matches": int(df["match_id"].nunique()),
        "avg_favorite_pre_match_odds": float(df["favorite_pre_match_odds"].mean()),
        "avg_serve_gap": float(df["favorite_set1_serve_gap"].mean()),
        "avg_hcap_line": float(df["set_hcap_line"].mean()),
        "avg_favorite_hcap_odds": float(df["favorite_set2_hcap_odds"].mean()),
        "cover_rate": cover_rate,
        "breakeven_rate": breakeven,
        "cover_edge": cover_rate - breakeven,
    }


def build_summary(rows: pd.DataFrame) -> pd.DataFrame:
    df = rows.copy()
    df["hcap_line_bucket"] = df["set_hcap_line"].apply(normalize_hcap_line)

    records: list[dict[str, float | int | str]] = []
    for source in ("opening", "first_seen"):
        source_rows = df[df["odds_source"] == source].copy()
        for line_bucket in sorted(source_rows["hcap_line_bucket"].dropna().unique()):
            scoped = source_rows[source_rows["hcap_line_bucket"] == line_bucket].copy()
            records.append(summarize_group(scoped, f"{source}_line_{line_bucket}"))

            strong_fav = scoped[scoped["favorite_pre_match_odds"] <= 1.50].copy()
            records.append(
                summarize_group(strong_fav, f"{source}_favorite_odds_le_1.50_line_{line_bucket}")
            )

            strong_serve = strong_fav[strong_fav["favorite_set1_serve_gap"] >= 0.08].copy()
            records.append(
                summarize_group(
                    strong_serve,
                    f"{source}_favorite_odds_le_1.50_serve_gap_ge_0.08_line_{line_bucket}",
                )
            )

    summary = pd.DataFrame(records)
    summary = summary[summary["samples"] >= MIN_SAMPLES].copy()
    summary = summary.sort_values(
        ["cover_edge", "samples"], ascending=[False, False]
    ).reset_index(drop=True)
    return summary


def main() -> None:
    rows = build_analysis_rows()
    hcap_rows = prepare_hcap_market(rows)
    summary = build_summary(hcap_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows_path = OUTPUT_DIR / "favorite_set1_set2_hcap_line_rows.csv"
    summary_path = OUTPUT_DIR / "favorite_set1_set2_hcap_line_summary.csv"
    hcap_rows.to_csv(rows_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("Favorite won set1 handicap line scan")
    print("Scenario: split set2 handicap by actual line buckets.")
    print(f"Rows: {len(hcap_rows)}")
    print(f"Matches: {hcap_rows['match_id'].nunique()}")

    for row in summary.head(30).itertuples(index=False):
        print(
            f"  {row.group}: samples={row.samples}, "
            f"cover_rate={row.cover_rate:.4f}, "
            f"breakeven={row.breakeven_rate:.4f}, "
            f"edge={row.cover_edge:.4f}, "
            f"hcap_line={row.avg_hcap_line:.2f}, "
            f"hcap_odds={row.avg_favorite_hcap_odds:.2f}, "
            f"fav_odds={row.avg_favorite_pre_match_odds:.2f}, "
            f"serve_gap={row.avg_serve_gap:.4f}"
        )

    print(f"\nSummary saved to {summary_path}")
    print(f"Prepared rows saved to {rows_path}")


if __name__ == "__main__":
    main()
