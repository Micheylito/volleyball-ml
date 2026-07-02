from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.favorite_set1_set2_market_analysis import build_analysis_rows


OUTPUT_DIR = Path("data/processed")
TARGET_MIN_ODDS = 1.43
TARGET_MAX_ODDS = 1.50
TARGET_MIN_SERVE_GAP = 0.08
MIN_GENERIC_SAMPLES = 80
MIN_LEAGUE_SAMPLES = 40
MIN_COUNTRY_SAMPLES = 60


def prepare_zone_rows() -> pd.DataFrame:
    rows = build_analysis_rows().copy()
    rows["favorite_set2_odds"] = pd.to_numeric(rows["favorite_set2_odds"], errors="coerce")
    rows["favorite_won_set2"] = pd.to_numeric(rows["favorite_won_set2"], errors="coerce")
    rows["favorite_set1_serve_gap"] = pd.to_numeric(
        rows["favorite_set1_serve_gap"], errors="coerce"
    )
    rows["best_of"] = pd.to_numeric(rows["best_of"], errors="coerce")

    rows = rows[
        (rows["favorite_set2_odds"] >= TARGET_MIN_ODDS)
        & (rows["favorite_set2_odds"] <= TARGET_MAX_ODDS)
        & (rows["favorite_set1_serve_gap"] >= TARGET_MIN_SERVE_GAP)
    ].copy()

    rows["set1_margin_for_favorite"] = 0.0
    home_favorite_mask = rows["favorite_team"] == 1
    away_favorite_mask = rows["favorite_team"] == 2

    rows.loc[home_favorite_mask, "set1_margin_for_favorite"] = (
        pd.to_numeric(rows.loc[home_favorite_mask, "set1_home_serve_wins"], errors="coerce").fillna(0.0)
        - pd.to_numeric(rows.loc[home_favorite_mask, "set1_away_serve_wins"], errors="coerce").fillna(0.0)
    )
    rows.loc[away_favorite_mask, "set1_margin_for_favorite"] = (
        pd.to_numeric(rows.loc[away_favorite_mask, "set1_away_serve_wins"], errors="coerce").fillna(0.0)
        - pd.to_numeric(rows.loc[away_favorite_mask, "set1_home_serve_wins"], errors="coerce").fillna(0.0)
    )
    rows["breakeven_rate"] = 1.0 / rows["favorite_set2_odds"]
    rows["value_edge"] = rows["favorite_won_set2"] - rows["breakeven_rate"]
    rows["zone_label"] = (
        f"{TARGET_MIN_ODDS:.2f}-{TARGET_MAX_ODDS:.2f}"
        f"__serve_gap_ge_{TARGET_MIN_SERVE_GAP:.2f}"
    )
    return rows


def summarize_frame(
    rows: pd.DataFrame,
    group_column: str,
    min_samples: int,
) -> pd.DataFrame:
    grouped = (
        rows.groupby(group_column, dropna=False)
        .agg(
            samples=("match_id", "count"),
            matches=("match_id", "nunique"),
            win_set2_rate=("favorite_won_set2", "mean"),
            avg_set2_odds=("favorite_set2_odds", "mean"),
            median_set2_odds=("favorite_set2_odds", "median"),
            avg_serve_gap=("favorite_set1_serve_gap", "mean"),
            avg_serve_pct=("favorite_set1_serve_pct", "mean"),
            avg_rally_number=("rally_number", "mean"),
            avg_score_total=("score1", "mean"),
            avg_breakeven_rate=("breakeven_rate", "mean"),
        )
        .reset_index()
    )
    grouped["avg_score_total"] = (
        rows.groupby(group_column, dropna=False)
        .apply(lambda frame: float((frame["score1"] + frame["score2"]).mean()))
        .values
    )
    grouped["edge_vs_breakeven"] = (
        grouped["win_set2_rate"] - grouped["avg_breakeven_rate"]
    )
    grouped = grouped[grouped["samples"] >= min_samples].copy()
    return grouped.sort_values(
        ["edge_vs_breakeven", "win_set2_rate", "samples"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def build_overall_summary(rows: pd.DataFrame) -> pd.DataFrame:
    overall = pd.DataFrame(
        [
            {
                "zone": rows["zone_label"].iloc[0] if not rows.empty else "",
                "samples": int(len(rows)),
                "matches": int(rows["match_id"].nunique()) if not rows.empty else 0,
                "win_set2_rate": float(rows["favorite_won_set2"].mean()) if not rows.empty else 0.0,
                "avg_set2_odds": float(rows["favorite_set2_odds"].mean()) if not rows.empty else 0.0,
                "median_set2_odds": float(rows["favorite_set2_odds"].median()) if not rows.empty else 0.0,
                "avg_serve_gap": float(rows["favorite_set1_serve_gap"].mean()) if not rows.empty else 0.0,
                "avg_serve_pct": float(rows["favorite_set1_serve_pct"].mean()) if not rows.empty else 0.0,
                "avg_breakeven_rate": float(rows["breakeven_rate"].mean()) if not rows.empty else 0.0,
                "edge_vs_breakeven": float(rows["favorite_won_set2"].mean() - rows["breakeven_rate"].mean())
                if not rows.empty
                else 0.0,
            }
        ]
    )
    return overall


def print_top_table(title: str, frame: pd.DataFrame, label_column: str, limit: int = 10) -> None:
    print(f"\n{title}:")
    if frame.empty:
        print("  no groups passed the sample filter")
        return

    for row in frame.head(limit).itertuples(index=False):
        label = getattr(row, label_column)
        print(
            f"  {label}: samples={row.samples}, "
            f"win_set2_rate={row.win_set2_rate:.4f}, "
            f"breakeven={row.avg_breakeven_rate:.4f}, "
            f"edge={row.edge_vs_breakeven:.4f}, "
            f"avg_odds={row.avg_set2_odds:.2f}, "
            f"serve_gap={row.avg_serve_gap:.4f}"
        )


def main() -> None:
    rows = prepare_zone_rows()
    overall = build_overall_summary(rows)
    odds_source_summary = summarize_frame(rows, "odds_source", MIN_GENERIC_SAMPLES)
    best_of_summary = summarize_frame(rows, "best_of", MIN_GENERIC_SAMPLES)
    gender_summary = summarize_frame(rows, "gender", MIN_GENERIC_SAMPLES)
    country_summary = summarize_frame(rows, "country", MIN_COUNTRY_SAMPLES)
    league_summary = summarize_frame(rows, "league", MIN_LEAGUE_SAMPLES)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows_path = OUTPUT_DIR / "favorite_set1_set2_market_zone_rows.csv"
    overall_path = OUTPUT_DIR / "favorite_set1_set2_market_zone_overall.csv"
    odds_source_path = OUTPUT_DIR / "favorite_set1_set2_market_zone_odds_source_summary.csv"
    best_of_path = OUTPUT_DIR / "favorite_set1_set2_market_zone_best_of_summary.csv"
    gender_path = OUTPUT_DIR / "favorite_set1_set2_market_zone_gender_summary.csv"
    country_path = OUTPUT_DIR / "favorite_set1_set2_market_zone_country_summary.csv"
    league_path = OUTPUT_DIR / "favorite_set1_set2_market_zone_league_summary.csv"

    rows.to_csv(rows_path, index=False)
    overall.to_csv(overall_path, index=False)
    odds_source_summary.to_csv(odds_source_path, index=False)
    best_of_summary.to_csv(best_of_path, index=False)
    gender_summary.to_csv(gender_path, index=False)
    country_summary.to_csv(country_path, index=False)
    league_summary.to_csv(league_path, index=False)

    print("Favorite set1 + set2 market zone scan")
    print(
        "Zone: "
        f"set2_odds {TARGET_MIN_ODDS:.2f}-{TARGET_MAX_ODDS:.2f}, "
        f"serve_gap >= {TARGET_MIN_SERVE_GAP:.2f}"
    )
    if not overall.empty:
        row = overall.iloc[0]
        print(
            f"Overall: samples={int(row['samples'])}, "
            f"win_set2_rate={row['win_set2_rate']:.4f}, "
            f"breakeven={row['avg_breakeven_rate']:.4f}, "
            f"edge={row['edge_vs_breakeven']:.4f}, "
            f"avg_odds={row['avg_set2_odds']:.2f}, "
            f"serve_gap={row['avg_serve_gap']:.4f}"
        )

    print_top_table("Odds source", odds_source_summary, "odds_source", limit=10)
    print_top_table("Best of", best_of_summary, "best_of", limit=10)
    print_top_table("Gender", gender_summary, "gender", limit=10)
    print_top_table("Top countries", country_summary, "country", limit=10)
    print_top_table("Top leagues", league_summary, "league", limit=15)

    print(f"\nRows saved to {rows_path}")
    print(f"Overall summary saved to {overall_path}")
    print(f"Odds source summary saved to {odds_source_path}")
    print(f"Best-of summary saved to {best_of_path}")
    print(f"Gender summary saved to {gender_path}")
    print(f"Country summary saved to {country_path}")
    print(f"League summary saved to {league_path}")


if __name__ == "__main__":
    main()
