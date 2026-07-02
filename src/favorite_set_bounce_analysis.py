from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.db import load_favorite_set_bounce_rows


OUTPUT_DIR = Path("data/processed")


def prepare_favorite_rows(rows: pd.DataFrame) -> pd.DataFrame:
    df = rows.copy()
    df["home_odds"] = pd.to_numeric(df["home_odds"], errors="coerce")
    df["away_odds"] = pd.to_numeric(df["away_odds"], errors="coerce")
    df["set1_winner"] = pd.to_numeric(df["set1_winner"], errors="coerce")
    df["set2_winner"] = pd.to_numeric(df["set2_winner"], errors="coerce")
    df["match_winner"] = pd.to_numeric(df["match_winner"], errors="coerce")

    df = df[
        df["home_odds"].notna()
        & df["away_odds"].notna()
        & df["set1_winner"].isin([1, 2])
        & df["match_winner"].isin([1, 2])
    ].copy()

    df["favorite_team"] = 0
    df.loc[df["home_odds"] < df["away_odds"], "favorite_team"] = 1
    df.loc[df["away_odds"] < df["home_odds"], "favorite_team"] = 2
    df = df[df["favorite_team"].isin([1, 2])].copy()

    df["favorite_lost_set1"] = (df["set1_winner"] != df["favorite_team"]).astype(int)
    df["favorite_won_set2"] = (df["set2_winner"] == df["favorite_team"]).astype(int)
    df["favorite_won_match"] = (df["match_winner"] == df["favorite_team"]).astype(int)
    df["favorite_side"] = df["favorite_team"].map({1: "home", 2: "away"})
    return df


def summarize_group(df: pd.DataFrame, label: str) -> dict[str, float | int | str]:
    return {
        "group": label,
        "samples": int(len(df)),
        "win_set2_rate": float(df["favorite_won_set2"].mean()) if not df.empty else 0.0,
        "win_match_rate": float(df["favorite_won_match"].mean()) if not df.empty else 0.0,
        "avg_home_odds": float(df["home_odds"].mean()) if not df.empty else 0.0,
        "avg_away_odds": float(df["away_odds"].mean()) if not df.empty else 0.0,
    }


def build_summary(prepared: pd.DataFrame) -> pd.DataFrame:
    all_favorites = prepared.copy()
    bounced = prepared[prepared["favorite_lost_set1"] == 1].copy()

    rows = [
        summarize_group(all_favorites, "all_favorites"),
        summarize_group(bounced, "favorite_lost_set1"),
    ]

    for odds_source in ("opening", "first_seen"):
        source_rows = bounced[bounced["odds_source"] == odds_source].copy()
        rows.append(summarize_group(source_rows, f"favorite_lost_set1_{odds_source}"))

    for favorite_side in ("home", "away"):
        side_rows = bounced[bounced["favorite_side"] == favorite_side].copy()
        rows.append(summarize_group(side_rows, f"favorite_lost_set1_{favorite_side}"))

    summary = pd.DataFrame(rows)
    baseline_set2 = float(summary.loc[summary["group"] == "all_favorites", "win_set2_rate"].iloc[0])
    baseline_match = float(summary.loc[summary["group"] == "all_favorites", "win_match_rate"].iloc[0])
    summary["set2_uplift_vs_all_favorites"] = summary["win_set2_rate"] - baseline_set2
    summary["match_uplift_vs_all_favorites"] = summary["win_match_rate"] - baseline_match
    return summary


def main() -> None:
    rows = load_favorite_set_bounce_rows()
    prepared = prepare_favorite_rows(rows)
    summary = build_summary(prepared)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prepared_path = OUTPUT_DIR / "favorite_set_bounce_rows.csv"
    summary_path = OUTPUT_DIR / "favorite_set_bounce_summary.csv"
    prepared.to_csv(prepared_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("Favorite set bounce analysis")
    for row in summary.itertuples(index=False):
        print(
            f"  {row.group}: samples={row.samples}, "
            f"win_set2_rate={row.win_set2_rate:.4f}, "
            f"win_match_rate={row.win_match_rate:.4f}, "
            f"set2_uplift={row.set2_uplift_vs_all_favorites:.4f}, "
            f"match_uplift={row.match_uplift_vs_all_favorites:.4f}"
        )
    print(f"Summary saved to {summary_path}")
    print(f"Prepared rows saved to {prepared_path}")


if __name__ == "__main__":
    main()
