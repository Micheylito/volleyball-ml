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
    return df


def build_current_set_live_features(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = prepare_current_set_live_frame(rows)
    x = df[BASELINE_SET_FEATURE_COLUMNS].fillna(0.0)
    y = df["target_set_team1_win"]
    return x, y
