from __future__ import annotations

from collections import defaultdict, deque

import pandas as pd


FORM_WINDOW = 5


def _average(values: deque[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _days_since(previous_date: pd.Timestamp | None, current_date: pd.Timestamp) -> float:
    if previous_date is None:
        return -1.0
    return float((current_date - previous_date).days)


def add_form_features(matches: pd.DataFrame) -> pd.DataFrame:
    df = matches.copy()

    recent_wins: dict[str, deque[int]] = defaultdict(lambda: deque(maxlen=FORM_WINDOW))
    recent_opponent_classes: dict[str, deque[float]] = defaultdict(
        lambda: deque(maxlen=FORM_WINDOW)
    )
    recent_dates: dict[str, pd.Timestamp | None] = defaultdict(lambda: None)

    home_recent_games: list[int] = []
    away_recent_games: list[int] = []
    home_recent_win_rate: list[float] = []
    away_recent_win_rate: list[float] = []
    home_recent_opp_class_avg: list[float] = []
    away_recent_opp_class_avg: list[float] = []
    home_days_since_last: list[float] = []
    away_days_since_last: list[float] = []

    for row in df.itertuples(index=False):
        match_date = row.match_date
        home_team = row.home_team
        away_team = row.away_team

        home_recent_games.append(len(recent_wins[home_team]))
        away_recent_games.append(len(recent_wins[away_team]))
        home_recent_win_rate.append(_average(recent_wins[home_team]))
        away_recent_win_rate.append(_average(recent_wins[away_team]))
        home_recent_opp_class_avg.append(_average(recent_opponent_classes[home_team]))
        away_recent_opp_class_avg.append(_average(recent_opponent_classes[away_team]))
        home_days_since_last.append(_days_since(recent_dates[home_team], match_date))
        away_days_since_last.append(_days_since(recent_dates[away_team], match_date))

        home_win = 1 if row.winner == 1 else 0
        away_win = 1 if row.winner == 2 else 0

        recent_wins[home_team].append(home_win)
        recent_wins[away_team].append(away_win)
        recent_opponent_classes[home_team].append(
            float(row.team2_class) if pd.notna(row.team2_class) else 0.0
        )
        recent_opponent_classes[away_team].append(
            float(row.team1_class) if pd.notna(row.team1_class) else 0.0
        )
        recent_dates[home_team] = match_date
        recent_dates[away_team] = match_date

    df["home_recent_games"] = home_recent_games
    df["away_recent_games"] = away_recent_games
    df["home_recent_win_rate"] = home_recent_win_rate
    df["away_recent_win_rate"] = away_recent_win_rate
    df["recent_win_rate_gap"] = df["home_recent_win_rate"] - df["away_recent_win_rate"]
    df["home_recent_opp_class_avg"] = home_recent_opp_class_avg
    df["away_recent_opp_class_avg"] = away_recent_opp_class_avg
    df["recent_opp_class_gap"] = (
        df["home_recent_opp_class_avg"] - df["away_recent_opp_class_avg"]
    )
    df["home_days_since_last"] = home_days_since_last
    df["away_days_since_last"] = away_days_since_last
    df["rest_days_gap"] = df["home_days_since_last"] - df["away_days_since_last"]
    return df


def build_features(matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = matches.copy()
    df["match_date"] = pd.to_datetime(df["match_date"])
    df = df.sort_values("match_date").reset_index(drop=True)
    df = add_form_features(df)

    df["target_home_win"] = (df["winner"] == 1).astype(int)
    df["odds_gap"] = df["away_odds"] - df["home_odds"]
    df["implied_home_prob"] = 1.0 / df["home_odds"]
    df["implied_away_prob"] = 1.0 / df["away_odds"]
    df["class_gap"] = df["team1_class"] - df["team2_class"]
    df["is_women"] = (df["gender"] == "W").astype(int)
    df["is_best_of_five"] = (df["best_of"] == 5).astype(int)
    df["odds_source_opening"] = (df["odds_source"] == "opening").astype(int)
    df["odds_source_first_seen"] = (df["odds_source"] == "first_seen").astype(int)

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
        "home_recent_games",
        "away_recent_games",
        "home_recent_win_rate",
        "away_recent_win_rate",
        "recent_win_rate_gap",
        "home_recent_opp_class_avg",
        "away_recent_opp_class_avg",
        "recent_opp_class_gap",
        "home_days_since_last",
        "away_days_since_last",
        "rest_days_gap",
        "odds_source_opening",
        "odds_source_first_seen",
    ]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    x = df[numeric_columns].fillna(0.0)
    y = df["target_home_win"]
    return x, y
