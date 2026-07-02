from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.favorite_set1_set2_side_markets_analysis import (
    build_analysis_rows,
    prepare_hcap_market,
)


OUTPUT_DIR = Path("data/processed")
TARGET_SOURCE = "first_seen"
TARGET_MAX_FAVORITE_ODDS = 1.50
TARGET_MIN_SERVE_GAP = 0.08
TARGET_HCAP_LINE = -3.5


def select_pattern_rows() -> pd.DataFrame:
    rows = build_analysis_rows()
    hcap_rows = prepare_hcap_market(rows).copy()
    hcap_rows["match_date"] = pd.to_datetime(hcap_rows["match_date"])

    selected = hcap_rows[
        (hcap_rows["odds_source"] == TARGET_SOURCE)
        & (hcap_rows["favorite_pre_match_odds"] <= TARGET_MAX_FAVORITE_ODDS)
        & (hcap_rows["favorite_set1_serve_gap"] >= TARGET_MIN_SERVE_GAP)
        & (hcap_rows["favorite_set2_hcap_line"].round(2) == TARGET_HCAP_LINE)
    ].copy()

    selected = selected.sort_values(["match_date", "match_id"]).reset_index(drop=True)
    return selected


def summarize_slice(df: pd.DataFrame, label: str) -> dict[str, float | int | str]:
    if df.empty:
        return {
            "slice": label,
            "samples": 0,
            "matches": 0,
            "cover_rate": 0.0,
            "breakeven": 0.0,
            "edge": 0.0,
            "avg_hcap_odds": 0.0,
            "avg_favorite_odds": 0.0,
            "avg_serve_gap": 0.0,
            "date_from": "",
            "date_to": "",
        }

    breakeven = float((1.0 / df["favorite_set2_hcap_odds"]).mean())
    cover_rate = float(df["favorite_set2_cover"].mean())
    return {
        "slice": label,
        "samples": int(len(df)),
        "matches": int(df["match_id"].nunique()),
        "cover_rate": cover_rate,
        "breakeven": breakeven,
        "edge": cover_rate - breakeven,
        "avg_hcap_odds": float(df["favorite_set2_hcap_odds"].mean()),
        "avg_favorite_odds": float(df["favorite_pre_match_odds"].mean()),
        "avg_serve_gap": float(df["favorite_set1_serve_gap"].mean()),
        "date_from": str(df["match_date"].min()),
        "date_to": str(df["match_date"].max()),
    }


def build_summary(rows: pd.DataFrame) -> pd.DataFrame:
    n = len(rows)
    if n == 0:
        return pd.DataFrame([summarize_slice(rows, "all")])

    split_50 = int(n * 0.50)
    split_70 = int(n * 0.70)
    split_80 = int(n * 0.80)

    parts = [
        ("all", rows),
        ("first_50pct", rows.iloc[:split_50].copy()),
        ("last_50pct", rows.iloc[split_50:].copy()),
        ("first_70pct", rows.iloc[:split_70].copy()),
        ("last_30pct", rows.iloc[split_70:].copy()),
        ("first_80pct", rows.iloc[:split_80].copy()),
        ("last_20pct", rows.iloc[split_80:].copy()),
    ]

    summary = pd.DataFrame([summarize_slice(frame, label) for label, frame in parts])
    return summary


def build_rolling_summary(rows: pd.DataFrame, chunk_size: int = 50) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(
            columns=[
                "chunk",
                "samples",
                "cover_rate",
                "breakeven",
                "edge",
                "avg_hcap_odds",
                "date_from",
                "date_to",
            ]
        )

    records: list[dict[str, float | int | str]] = []
    chunk_number = 1
    for start in range(0, len(rows), chunk_size):
        chunk = rows.iloc[start : start + chunk_size].copy()
        breakeven = float((1.0 / chunk["favorite_set2_hcap_odds"]).mean())
        cover_rate = float(chunk["favorite_set2_cover"].mean())
        records.append(
            {
                "chunk": chunk_number,
                "samples": int(len(chunk)),
                "cover_rate": cover_rate,
                "breakeven": breakeven,
                "edge": cover_rate - breakeven,
                "avg_hcap_odds": float(chunk["favorite_set2_hcap_odds"].mean()),
                "date_from": str(chunk["match_date"].min()),
                "date_to": str(chunk["match_date"].max()),
            }
        )
        chunk_number += 1

    return pd.DataFrame(records)


def main() -> None:
    rows = select_pattern_rows()
    summary = build_summary(rows)
    rolling = build_rolling_summary(rows, chunk_size=50)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows_path = OUTPUT_DIR / "favorite_set1_hcap_chronological_rows.csv"
    summary_path = OUTPUT_DIR / "favorite_set1_hcap_chronological_summary.csv"
    rolling_path = OUTPUT_DIR / "favorite_set1_hcap_chronological_rolling.csv"

    rows.to_csv(rows_path, index=False)
    summary.to_csv(summary_path, index=False)
    rolling.to_csv(rolling_path, index=False)

    print("Favorite won set1 handicap chronological validation")
    print(
        "Pattern: "
        f"{TARGET_SOURCE}, favorite_odds <= {TARGET_MAX_FAVORITE_ODDS:.2f}, "
        f"serve_gap >= {TARGET_MIN_SERVE_GAP:.2f}, "
        f"hcap_line = {TARGET_HCAP_LINE:.1f}"
    )
    print(f"Rows: {len(rows)}")
    print(f"Matches: {rows['match_id'].nunique() if not rows.empty else 0}")

    for row in summary.itertuples(index=False):
        print(
            f"  {row.slice}: samples={row.samples}, "
            f"cover_rate={row.cover_rate:.4f}, "
            f"breakeven={row.breakeven:.4f}, "
            f"edge={row.edge:.4f}, "
            f"avg_odds={row.avg_hcap_odds:.2f}, "
            f"date_from={row.date_from}, date_to={row.date_to}"
        )

    if not rolling.empty:
        print("\nRolling chunks (50 matches each):")
        for row in rolling.itertuples(index=False):
            print(
                f"  chunk_{row.chunk}: samples={row.samples}, "
                f"cover_rate={row.cover_rate:.4f}, "
                f"breakeven={row.breakeven:.4f}, "
                f"edge={row.edge:.4f}, "
                f"avg_odds={row.avg_hcap_odds:.2f}"
            )

    print(f"\nSummary saved to {summary_path}")
    print(f"Rolling summary saved to {rolling_path}")
    print(f"Rows saved to {rows_path}")


if __name__ == "__main__":
    main()
