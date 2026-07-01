from __future__ import annotations

from collections import defaultdict, deque

import pandas as pd


FORM_WINDOWS = (3, 5, 10)


def _average(values: deque[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _days_since(previous_date: pd.Timestamp | None, current_date: pd.Timestamp) -> float:
    if previous_date is None:
        return -1.0
    return float((current_date - previous_date).days)


def add_form_features(matches: pd.DataFrame) -> pd.DataFrame:
    df = matches.copy()

    recent_wins = {
        window: defaultdict(lambda: deque(maxlen=window)) for window in FORM_WINDOWS
    }
    recent_opponent_classes = {
        window: defaultdict(lambda: deque(maxlen=window)) for window in FORM_WINDOWS
    }
    recent_weighted_wins = {
        window: defaultdict(lambda: deque(maxlen=window)) for window in FORM_WINDOWS
    }
    recent_strength_of_schedule = {
        window: defaultdict(lambda: deque(maxlen=window)) for window in FORM_WINDOWS
    }
    recent_dates: dict[str, pd.Timestamp | None] = defaultdict(lambda: None)

    form_values: dict[str, list[float]] = {}
    for window in FORM_WINDOWS:
        form_values[f"home_recent_games_{window}"] = []
        form_values[f"away_recent_games_{window}"] = []
        form_values[f"home_recent_win_rate_{window}"] = []
        form_values[f"away_recent_win_rate_{window}"] = []
        form_values[f"home_recent_opp_class_avg_{window}"] = []
        form_values[f"away_recent_opp_class_avg_{window}"] = []
        form_values[f"home_recent_weighted_win_score_{window}"] = []
        form_values[f"away_recent_weighted_win_score_{window}"] = []
        form_values[f"home_recent_schedule_strength_{window}"] = []
        form_values[f"away_recent_schedule_strength_{window}"] = []
    home_days_since_last: list[float] = []
    away_days_since_last: list[float] = []

    for row in df.itertuples(index=False):
        match_date = row.match_date
        home_team = row.home_team
        away_team = row.away_team
        home_opponent_class = float(row.team2_class) if pd.notna(row.team2_class) else 0.0
        away_opponent_class = float(row.team1_class) if pd.notna(row.team1_class) else 0.0

        for window in FORM_WINDOWS:
            form_values[f"home_recent_games_{window}"].append(len(recent_wins[window][home_team]))
            form_values[f"away_recent_games_{window}"].append(len(recent_wins[window][away_team]))
            form_values[f"home_recent_win_rate_{window}"].append(
                _average(recent_wins[window][home_team])
            )
            form_values[f"away_recent_win_rate_{window}"].append(
                _average(recent_wins[window][away_team])
            )
            form_values[f"home_recent_opp_class_avg_{window}"].append(
                _average(recent_opponent_classes[window][home_team])
            )
            form_values[f"away_recent_opp_class_avg_{window}"].append(
                _average(recent_opponent_classes[window][away_team])
            )
            form_values[f"home_recent_weighted_win_score_{window}"].append(
                _average(recent_weighted_wins[window][home_team])
            )
            form_values[f"away_recent_weighted_win_score_{window}"].append(
                _average(recent_weighted_wins[window][away_team])
            )
            form_values[f"home_recent_schedule_strength_{window}"].append(
                _average(recent_strength_of_schedule[window][home_team])
            )
            form_values[f"away_recent_schedule_strength_{window}"].append(
                _average(recent_strength_of_schedule[window][away_team])
            )
        home_days_since_last.append(_days_since(recent_dates[home_team], match_date))
        away_days_since_last.append(_days_since(recent_dates[away_team], match_date))

        home_win = 1 if row.winner == 1 else 0
        away_win = 1 if row.winner == 2 else 0
        home_weighted_win = home_win * (1.0 + home_opponent_class)
        away_weighted_win = away_win * (1.0 + away_opponent_class)

        for window in FORM_WINDOWS:
            recent_wins[window][home_team].append(home_win)
            recent_wins[window][away_team].append(away_win)
            recent_opponent_classes[window][home_team].append(home_opponent_class)
            recent_opponent_classes[window][away_team].append(away_opponent_class)
            recent_weighted_wins[window][home_team].append(home_weighted_win)
            recent_weighted_wins[window][away_team].append(away_weighted_win)
            recent_strength_of_schedule[window][home_team].append(home_opponent_class)
            recent_strength_of_schedule[window][away_team].append(away_opponent_class)
        recent_dates[home_team] = match_date
        recent_dates[away_team] = match_date

    for column_name, values in form_values.items():
        df[column_name] = values

    for window in FORM_WINDOWS:
        df[f"recent_win_rate_gap_{window}"] = (
            df[f"home_recent_win_rate_{window}"] - df[f"away_recent_win_rate_{window}"]
        )
        df[f"recent_opp_class_gap_{window}"] = (
            df[f"home_recent_opp_class_avg_{window}"]
            - df[f"away_recent_opp_class_avg_{window}"]
        )
        df[f"recent_weighted_win_score_gap_{window}"] = (
            df[f"home_recent_weighted_win_score_{window}"]
            - df[f"away_recent_weighted_win_score_{window}"]
        )
        df[f"recent_schedule_strength_gap_{window}"] = (
            df[f"home_recent_schedule_strength_{window}"]
            - df[f"away_recent_schedule_strength_{window}"]
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
        "home_days_since_last",
        "away_days_since_last",
        "rest_days_gap",
        "odds_source_opening",
        "odds_source_first_seen",
    ]
    for window in FORM_WINDOWS:
        numeric_columns.extend(
            [
                f"home_recent_games_{window}",
                f"away_recent_games_{window}",
                f"home_recent_win_rate_{window}",
                f"away_recent_win_rate_{window}",
                f"recent_win_rate_gap_{window}",
                f"home_recent_opp_class_avg_{window}",
                f"away_recent_opp_class_avg_{window}",
                f"recent_opp_class_gap_{window}",
                f"home_recent_weighted_win_score_{window}",
                f"away_recent_weighted_win_score_{window}",
                f"recent_weighted_win_score_gap_{window}",
                f"home_recent_schedule_strength_{window}",
                f"away_recent_schedule_strength_{window}",
                f"recent_schedule_strength_gap_{window}",
            ]
        )
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    x = df[numeric_columns].fillna(0.0)
    y = df["target_home_win"]
    return x, y
