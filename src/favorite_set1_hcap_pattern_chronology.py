from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.favorite_set1_set2_side_markets_analysis import (
    build_analysis_rows,
    prepare_hcap_market,
)


OUTPUT_DIR = Path("data/processed")
TARGET_HCAP_LINE = -3.5
ROLLING_CHUNK_SIZE = 50

PATTERNS = (
    {
        "name": "opening_135_150_pct60_gap10",
        "source": "opening",
        "min_odds": 1.35,
        "max_odds": 1.50,
        "min_serve_pct": 0.60,
        "min_serve_gap": 0.10,
    },
    {
        "name": "first_seen_115_120_pct55_gap10",
        "source": "first_seen",
        "min_odds": 1.15,
        "max_odds": 1.20,
        "min_serve_pct": 0.55,
        "min_serve_gap": 0.10,
    },
    {
        "name": "first_seen_120_135_pct55_gap05",
        "source": "first_seen",
        "min_odds": 1.20,
        "max_odds": 1.35,
        "min_serve_pct": 0.55,
        "min_serve_gap": 0.05,
    },
)


def load_hcap_rows() -> pd.DataFrame:
    rows = build_analysis_rows()
    hcap_rows = prepare_hcap_market(rows).copy()
    hcap_rows["match_date"] = pd.to_datetime(hcap_rows["match_date"])
    selected = hcap_rows[
        hcap_rows["favorite_set2_hcap_line"].round(2) == TARGET_HCAP_LINE
    ].copy()
    selected = selected.sort_values(["match_date", "match_id"]).reset_index(drop=True)
    return selected


def select_pattern_rows(rows: pd.DataFrame, pattern: dict[str, float | str]) -> pd.DataFrame:
    selected = rows[
        (rows["odds_source"] == pattern["source"])
        & (rows["favorite_pre_match_odds"] >= pattern["min_odds"])
        & (rows["favorite_pre_match_odds"] < pattern["max_odds"])
        & (rows["favorite_set1_serve_pct"] >= pattern["min_serve_pct"])
        & (rows["favorite_set1_serve_gap"] >= pattern["min_serve_gap"])
    ].copy()
    selected = selected.sort_values(["match_date", "match_id"]).reset_index(drop=True)
    return selected


def summarize_slice(
    df: pd.DataFrame,
    *,
    pattern_name: str,
    slice_name: str,
) -> dict[str, float | int | str]:
    if df.empty:
        return {
            "pattern": pattern_name,
            "slice": slice_name,
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
        "pattern": pattern_name,
        "slice": slice_name,
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


def build_split_summary(rows: pd.DataFrame, pattern_name: str) -> pd.DataFrame:
    n = len(rows)
    if n == 0:
        return pd.DataFrame([summarize_slice(rows, pattern_name=pattern_name, slice_name="all")])

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
    return pd.DataFrame(
        [
            summarize_slice(frame, pattern_name=pattern_name, slice_name=slice_name)
            for slice_name, frame in parts
        ]
    )


def build_rolling_summary(rows: pd.DataFrame, pattern_name: str) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(
            columns=[
                "pattern",
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
    for start in range(0, len(rows), ROLLING_CHUNK_SIZE):
        chunk = rows.iloc[start : start + ROLLING_CHUNK_SIZE].copy()
        breakeven = float((1.0 / chunk["favorite_set2_hcap_odds"]).mean())
        cover_rate = float(chunk["favorite_set2_cover"].mean())
        records.append(
            {
                "pattern": pattern_name,
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
    base_rows = load_hcap_rows()

    split_frames: list[pd.DataFrame] = []
    rolling_frames: list[pd.DataFrame] = []
    selected_frames: list[pd.DataFrame] = []

    print("Favorite won set1 handicap pattern chronology")
    print(f"Target handicap line: {TARGET_HCAP_LINE:.1f}")
    print(f"Rolling chunk size: {ROLLING_CHUNK_SIZE}")

    for pattern in PATTERNS:
        pattern_rows = select_pattern_rows(base_rows, pattern)
        pattern_name = str(pattern["name"])
        selected_frame = pattern_rows.copy()
        selected_frame["pattern"] = pattern_name
        selected_frames.append(selected_frame)

        split_summary = build_split_summary(pattern_rows, pattern_name)
        rolling_summary = build_rolling_summary(pattern_rows, pattern_name)
        split_frames.append(split_summary)
        rolling_frames.append(rolling_summary)

        print(f"\nPattern: {pattern_name}")
        print(
            f"  source={pattern['source']}, "
            f"odds={pattern['min_odds']:.2f}-{pattern['max_odds']:.2f}, "
            f"serve_pct>={pattern['min_serve_pct']:.2f}, "
            f"serve_gap>={pattern['min_serve_gap']:.2f}"
        )

        all_row = split_summary[split_summary["slice"] == "all"].iloc[0]
        print(
            f"  all: samples={int(all_row['samples'])}, "
            f"cover_rate={all_row['cover_rate']:.4f}, "
            f"breakeven={all_row['breakeven']:.4f}, "
            f"edge={all_row['edge']:.4f}, "
            f"avg_odds={all_row['avg_hcap_odds']:.2f}"
        )

        for slice_name in ("first_50pct", "last_50pct", "first_80pct", "last_20pct"):
            row = split_summary[split_summary["slice"] == slice_name].iloc[0]
            print(
                f"  {slice_name}: samples={int(row['samples'])}, "
                f"cover_rate={row['cover_rate']:.4f}, "
                f"breakeven={row['breakeven']:.4f}, "
                f"edge={row['edge']:.4f}"
            )

    split_summary_all = pd.concat(split_frames, ignore_index=True)
    rolling_summary_all = pd.concat(rolling_frames, ignore_index=True)
    selected_rows_all = pd.concat(selected_frames, ignore_index=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    split_path = OUTPUT_DIR / "favorite_set1_hcap_pattern_chronology_summary.csv"
    rolling_path = OUTPUT_DIR / "favorite_set1_hcap_pattern_chronology_rolling.csv"
    rows_path = OUTPUT_DIR / "favorite_set1_hcap_pattern_chronology_rows.csv"

    split_summary_all.to_csv(split_path, index=False)
    rolling_summary_all.to_csv(rolling_path, index=False)
    selected_rows_all.to_csv(rows_path, index=False)

    print(f"\nSummary saved to {split_path}")
    print(f"Rolling summary saved to {rolling_path}")
    print(f"Rows saved to {rows_path}")


if __name__ == "__main__":
    main()
