from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.live_set_db import load_serve_streak_rows


OUTPUT_DIR = Path("data/processed")


def prepare_serve_streak_frame(rows: pd.DataFrame) -> pd.DataFrame:
    df = rows.copy()
    df["rally_ts"] = pd.to_datetime(df["rally_ts"])
    df["set_number"] = pd.to_numeric(df["set_number"], errors="coerce")
    df["rally_number"] = pd.to_numeric(df["rally_number"], errors="coerce")
    df["score1"] = pd.to_numeric(df["score1"], errors="coerce")
    df["score2"] = pd.to_numeric(df["score2"], errors="coerce")
    df = df[
        df["set_number"].notna()
        & df["rally_number"].notna()
        & df["score1"].notna()
        & df["score2"].notna()
    ].copy()

    df = df.sort_values(
        ["match_id", "set_number", "rally_number", "rally_db_id"]
    ).reset_index(drop=True)

    streak_before_rally: list[int] = []
    inferred_point_winners: list[int | None] = []
    inferred_serve_teams: list[int | None] = []
    current_group: tuple[int, int] | None = None
    current_server = 0
    current_streak = 0
    previous_score1: float | None = None
    previous_score2: float | None = None
    previous_point_winner = 0

    for row in df.itertuples(index=False):
        group_key = (int(row.match_id), int(row.set_number))
        score1 = float(row.score1)
        score2 = float(row.score2)

        if group_key != current_group:
            current_group = group_key
            current_server = 0
            current_streak = 0
            previous_score1 = None
            previous_score2 = None
            previous_point_winner = 0

        point_winner: int | None = None
        if previous_score1 is not None and previous_score2 is not None:
            delta1 = score1 - previous_score1
            delta2 = score2 - previous_score2
            if delta1 == 1 and delta2 == 0:
                point_winner = 1
            elif delta1 == 0 and delta2 == 1:
                point_winner = 2

        serve_team: int | None = previous_point_winner if previous_point_winner in (1, 2) else None

        inferred_point_winners.append(point_winner)
        inferred_serve_teams.append(serve_team)

        if serve_team in (1, 2):
            if serve_team == current_server:
                current_streak += 1
            else:
                current_server = serve_team
                current_streak = 1
            streak_before_rally.append(current_streak)
        else:
            current_server = 0
            current_streak = 0
            streak_before_rally.append(0)

        previous_score1 = score1
        previous_score2 = score2
        previous_point_winner = point_winner or 0

    df["inferred_point_winner"] = inferred_point_winners
    df["inferred_serve_team"] = inferred_serve_teams
    df["serve_streak_before_rally"] = streak_before_rally
    df = df[
        df["inferred_point_winner"].isin([1, 2])
        & df["inferred_serve_team"].isin([1, 2])
        & (df["serve_streak_before_rally"] > 0)
    ].copy()

    if df.empty:
        raise ValueError("No valid rallies left after inferring serve streak inputs from score progression.")

    df["server_won_rally"] = (df["inferred_serve_team"] == df["inferred_point_winner"]).astype(int)
    df["set_score_gap"] = (df["score1"] - df["score2"]).abs()
    df["set_total_points"] = df["score1"] + df["score2"]
    df["is_clutch_phase"] = ((df["score1"] >= 20) | (df["score2"] >= 20)).astype(int)

    df["serve_streak_bucket"] = df["serve_streak_before_rally"].map(
        lambda value: f"{value}" if value < 5 else "5+"
    )
    return df


def build_bucket_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("serve_streak_bucket", dropna=False)
        .agg(
            rallies=("rally_db_id", "count"),
            server_win_rate=("server_won_rally", "mean"),
            avg_total_points=("set_total_points", "mean"),
            clutch_share=("is_clutch_phase", "mean"),
        )
        .reset_index()
    )

    bucket_order = {str(value): value for value in range(1, 5)}
    bucket_order["5+"] = 5
    summary["bucket_sort"] = summary["serve_streak_bucket"].map(bucket_order).fillna(999)
    summary = summary.sort_values("bucket_sort").drop(columns=["bucket_sort"]).reset_index(drop=True)
    return summary


def build_split_summary(df: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    splits = [
        ("all", df),
        ("clutch_only", df[df["is_clutch_phase"] == 1].copy()),
        ("non_clutch", df[df["is_clutch_phase"] == 0].copy()),
        ("close_score_gap_le_2", df[df["set_score_gap"] <= 2].copy()),
    ]

    for split_name, split_df in splits:
        if split_df.empty:
            continue
        summary = build_bucket_summary(split_df)
        summary.insert(0, "split", split_name)
        frames.append(summary)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    rows = load_serve_streak_rows()
    prepared = prepare_serve_streak_frame(rows)
    summary = build_split_summary(prepared)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows_path = OUTPUT_DIR / "serve_streak_analysis_rows.csv"
    summary_path = OUTPUT_DIR / "serve_streak_analysis_summary.csv"

    prepared.to_csv(rows_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("Serve streak analysis")
    print(f"Loaded rallies: {len(prepared)}")
    print("\nServer win rate by serve streak before rally:")
    for row in summary[summary["split"] == "all"].itertuples(index=False):
        print(
            f"  streak={row.serve_streak_bucket}: "
            f"rallies={row.rallies}, "
            f"server_win_rate={row.server_win_rate:.4f}, "
            f"clutch_share={row.clutch_share:.4f}"
        )

    print(f"\nDetailed summary saved to {summary_path}")
    print(f"Prepared rows saved to {rows_path}")


if __name__ == "__main__":
    main()
