from __future__ import annotations

import pandas as pd


BASELINE_SET_FEATURE_COLUMNS = [
    "set_win1",
    "set_win2",
    "set_number",
    "score1",
    "score2",
    "serve_team",
    "set_score_gap",
    "set_total_points",
]

EXTENDED_SET_FEATURE_COLUMNS = [
    *BASELINE_SET_FEATURE_COLUMNS,
    "is_clutch_phase",
    "points_to_25_home",
    "points_to_25_away",
    "is_two_point_endgame",
    "serve_score_interaction",
]


def prepare_current_set_live_frame(rows: pd.DataFrame) -> pd.DataFrame:
    df = rows.copy()
    df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"])
    df = df.sort_values("snapshot_ts").reset_index(drop=True)

    df["set_win1"] = pd.to_numeric(df["set_win1"], errors="coerce")
    df["set_win2"] = pd.to_numeric(df["set_win2"], errors="coerce")
    df["set_number"] = pd.to_numeric(df["set_number"], errors="coerce")
    df["score1"] = pd.to_numeric(df["score1"], errors="coerce").fillna(0.0)
    df["score2"] = pd.to_numeric(df["score2"], errors="coerce").fillna(0.0)
    df["serve_team"] = pd.to_numeric(df["serve_team"], errors="coerce").fillna(0.0)
    df["target_set_team1_win"] = pd.to_numeric(
        df["target_set_team1_win"], errors="coerce"
    ).fillna(0).astype(int)

    df["set_score_gap"] = df["score1"] - df["score2"]
    df["set_total_points"] = df["score1"] + df["score2"]
    df["points_to_25_home"] = (25.0 - df["score1"]).clip(lower=0.0)
    df["points_to_25_away"] = (25.0 - df["score2"]).clip(lower=0.0)
    df["is_clutch_phase"] = (
        (df["score1"] >= 20.0) | (df["score2"] >= 20.0)
    ).astype(int)
    df["is_two_point_endgame"] = (
        (df["score1"] >= 24.0) | (df["score2"] >= 24.0)
    ).astype(int)
    df["serve_score_interaction"] = df["serve_team"] * df["set_score_gap"]
    return df


def build_current_set_live_features(
    rows: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    df = prepare_current_set_live_frame(rows)
    selected_columns = feature_columns or BASELINE_SET_FEATURE_COLUMNS
    x = df[selected_columns].fillna(0.0)
    y = df["target_set_team1_win"]
    return x, y
