from __future__ import annotations

import pandas as pd


def build_features(matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = matches.copy()
    df["match_date"] = pd.to_datetime(df["match_date"])
    df = df.sort_values("match_date").reset_index(drop=True)

    df["target_home_win"] = (df["home_sets"] > df["away_sets"]).astype(int)
    df["odds_gap"] = df["away_odds"] - df["home_odds"]
    df["total_sets"] = df["home_sets"] + df["away_sets"]

    feature_columns = ["home_odds", "away_odds", "odds_gap", "total_sets"]
    x = df[feature_columns].fillna(0.0)
    y = df["target_home_win"]
    return x, y

