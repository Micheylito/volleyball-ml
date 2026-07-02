from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.db import load_favorite_set1_serve_rows


OUTPUT_DIR = Path("data/processed")
SERVE_PCT_THRESHOLDS = (0.50, 0.55, 0.60)
SERVE_GAP_THRESHOLDS = (0.03, 0.05, 0.08)


def prepare_rows(rows: pd.DataFrame) -> pd.DataFrame:
    df = rows.copy()
    numeric_columns = [
        "home_odds",
        "away_odds",
        "set1_winner",
        "set2_winner",
        "set3_winner",
        "match_winner",
        "set1_home_serve_pct",
        "set1_away_serve_pct",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

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

    df["favorite_won_set1"] = (df["set1_winner"] == df["favorite_team"]).astype(int)
    df["favorite_won_set2"] = (df["set2_winner"] == df["favorite_team"]).astype(int)
    df["favorite_won_set3"] = (df["set3_winner"] == df["favorite_team"]).astype(int)
    df["favorite_won_match"] = (df["match_winner"] == df["favorite_team"]).astype(int)

    df["favorite_set1_serve_pct"] = df["set1_home_serve_pct"]
    df["opponent_set1_serve_pct"] = df["set1_away_serve_pct"]
    away_favorite_mask = df["favorite_team"] == 2
    df.loc[away_favorite_mask, "favorite_set1_serve_pct"] = df.loc[
        away_favorite_mask, "set1_away_serve_pct"
    ]
    df.loc[away_favorite_mask, "opponent_set1_serve_pct"] = df.loc[
        away_favorite_mask, "set1_home_serve_pct"
    ]
    df["favorite_set1_serve_gap"] = (
        df["favorite_set1_serve_pct"] - df["opponent_set1_serve_pct"]
    )
    return df


def summarize_group(df: pd.DataFrame, label: str) -> dict[str, float | int | str]:
    set3_available = df["set3_winner"].isin([1, 2]) if not df.empty else pd.Series(dtype=bool)
    return {
        "group": label,
        "samples": int(len(df)),
        "win_set2_rate": float(df["favorite_won_set2"].mean()) if not df.empty else 0.0,
        "win_set3_rate": float(df.loc[set3_available, "favorite_won_set3"].mean())
        if not df.empty and set3_available.any()
        else 0.0,
        "set3_samples": int(set3_available.sum()) if not df.empty else 0,
        "win_match_rate": float(df["favorite_won_match"].mean()) if not df.empty else 0.0,
        "avg_favorite_set1_serve_pct": float(df["favorite_set1_serve_pct"].mean())
        if not df.empty
        else 0.0,
        "avg_favorite_set1_serve_gap": float(df["favorite_set1_serve_gap"].mean())
        if not df.empty
        else 0.0,
    }


def build_summary(prepared: pd.DataFrame) -> pd.DataFrame:
    favorite_won_set1 = prepared[prepared["favorite_won_set1"] == 1].copy()
    rows = [summarize_group(favorite_won_set1, "favorite_won_set1")]

    for threshold in SERVE_PCT_THRESHOLDS:
        scoped = favorite_won_set1[favorite_won_set1["favorite_set1_serve_pct"] >= threshold].copy()
        rows.append(summarize_group(scoped, f"favorite_won_set1_serve_pct_ge_{threshold:.2f}"))

    for threshold in SERVE_GAP_THRESHOLDS:
        scoped = favorite_won_set1[favorite_won_set1["favorite_set1_serve_gap"] >= threshold].copy()
        rows.append(summarize_group(scoped, f"favorite_won_set1_serve_gap_ge_{threshold:.2f}"))

    summary = pd.DataFrame(rows)
    baseline_set2 = float(summary.loc[summary["group"] == "favorite_won_set1", "win_set2_rate"].iloc[0])
    baseline_set3 = float(summary.loc[summary["group"] == "favorite_won_set1", "win_set3_rate"].iloc[0])
    baseline_match = float(summary.loc[summary["group"] == "favorite_won_set1", "win_match_rate"].iloc[0])
    summary["set2_uplift_vs_favorite_won_set1"] = summary["win_set2_rate"] - baseline_set2
    summary["set3_uplift_vs_favorite_won_set1"] = summary["win_set3_rate"] - baseline_set3
    summary["match_uplift_vs_favorite_won_set1"] = summary["win_match_rate"] - baseline_match
    return summary


def main() -> None:
    rows = load_favorite_set1_serve_rows()
    prepared = prepare_rows(rows)
    summary = build_summary(prepared)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prepared_path = OUTPUT_DIR / "favorite_set1_serve_rows.csv"
    summary_path = OUTPUT_DIR / "favorite_set1_serve_summary.csv"
    prepared.to_csv(prepared_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("Favorite set1 serve analysis")
    for row in summary.itertuples(index=False):
        print(
            f"  {row.group}: samples={row.samples}, "
            f"win_set2_rate={row.win_set2_rate:.4f}, "
            f"win_set3_rate={row.win_set3_rate:.4f} (n={row.set3_samples}), "
            f"win_match_rate={row.win_match_rate:.4f}, "
            f"avg_serve_pct={row.avg_favorite_set1_serve_pct:.4f}, "
            f"avg_serve_gap={row.avg_favorite_set1_serve_gap:.4f}, "
            f"set2_uplift={row.set2_uplift_vs_favorite_won_set1:.4f}, "
            f"set3_uplift={row.set3_uplift_vs_favorite_won_set1:.4f}, "
            f"match_uplift={row.match_uplift_vs_favorite_won_set1:.4f}"
        )
    print(f"Summary saved to {summary_path}")
    print(f"Prepared rows saved to {prepared_path}")


if __name__ == "__main__":
    main()
