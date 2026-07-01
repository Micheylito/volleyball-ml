from __future__ import annotations

import pandas as pd


def build_features(matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = matches.copy()
    df["match_date"] = pd.to_datetime(df["match_date"])
    df = df.sort_values("match_date").reset_index(drop=True)

    df["target_home_win"] = (df["winner"] == 1).astype(int)
    df["odds_gap"] = df["away_odds"] - df["home_odds"]
    df["implied_home_prob"] = 1.0 / df["home_odds"]
    df["implied_away_prob"] = 1.0 / df["away_odds"]
    df["class_gap"] = df["team1_class"] - df["team2_class"]
    df["is_women"] = (df["gender"] == "W").astype(int)
    df["is_best_of_five"] = (df["best_of"] == 5).astype(int)

    numeric_columns = [
        "home_odds",
        "away_odds",
        "odds_gap",
        "implied_home_prob",
        "implied_away_prob",
        "team1_class",
        "team2_class",
        "class_gap",
        "match_class",
        "best_of",
        "match_total_line",
        "match_total_over",
        "match_total_under",
        "set1_win1",
        "set1_win2",
        "set1_total_line",
        "set1_total_over",
        "set1_total_under",
        "is_women",
        "is_best_of_five",
    ]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    x = df[numeric_columns].fillna(0.0)
    y = df["target_home_win"]
    return x, y
