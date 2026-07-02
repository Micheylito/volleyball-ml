from __future__ import annotations

from collections import defaultdict, deque

import pandas as pd


FORM_WINDOWS = (3, 5, 10)
BASE_FEATURE_COLUMNS = [
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
    "live_home_serve_pct",
    "live_away_serve_pct",
    "live_serve_pct_gap",
    "live_home_serve_volume",
    "live_away_serve_volume",
    "live_serve_volume_gap",
]


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
    recent_serve_pct = {
        window: defaultdict(lambda: deque(maxlen=window)) for window in FORM_WINDOWS
    }
    recent_serve_volume = {
        window: defaultdict(lambda: deque(maxlen=window)) for window in FORM_WINDOWS
    }
    league_recent_wins = {
        window: defaultdict(lambda: deque(maxlen=window)) for window in FORM_WINDOWS
    }
    league_recent_serve_pct = {
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
        form_values[f"home_recent_serve_pct_{window}"] = []
        form_values[f"away_recent_serve_pct_{window}"] = []
        form_values[f"home_recent_serve_volume_{window}"] = []
        form_values[f"away_recent_serve_volume_{window}"] = []
        form_values[f"home_league_recent_win_rate_{window}"] = []
        form_values[f"away_league_recent_win_rate_{window}"] = []
        form_values[f"home_league_recent_serve_pct_{window}"] = []
        form_values[f"away_league_recent_serve_pct_{window}"] = []
    home_days_since_last: list[float] = []
    away_days_since_last: list[float] = []

    for row in df.itertuples(index=False):
        match_date = row.match_date
        home_team = row.home_team
        away_team = row.away_team
        league = row.league if pd.notna(row.league) else "unknown"
        home_league_key = (home_team, league)
        away_league_key = (away_team, league)
        home_opponent_class = float(row.team2_class) if pd.notna(row.team2_class) else 0.0
        away_opponent_class = float(row.team1_class) if pd.notna(row.team1_class) else 0.0
        home_match_serve_pct = (
            float(row.home_match_serve_pct) if pd.notna(row.home_match_serve_pct) else 0.0
        )
        away_match_serve_pct = (
            float(row.away_match_serve_pct) if pd.notna(row.away_match_serve_pct) else 0.0
        )
        home_match_serves = float(row.home_match_serves) if pd.notna(row.home_match_serves) else 0.0
        away_match_serves = float(row.away_match_serves) if pd.notna(row.away_match_serves) else 0.0

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
            form_values[f"home_recent_serve_pct_{window}"].append(
                _average(recent_serve_pct[window][home_team])
            )
            form_values[f"away_recent_serve_pct_{window}"].append(
                _average(recent_serve_pct[window][away_team])
            )
            form_values[f"home_recent_serve_volume_{window}"].append(
                _average(recent_serve_volume[window][home_team])
            )
            form_values[f"away_recent_serve_volume_{window}"].append(
                _average(recent_serve_volume[window][away_team])
            )
            form_values[f"home_league_recent_win_rate_{window}"].append(
                _average(league_recent_wins[window][home_league_key])
            )
            form_values[f"away_league_recent_win_rate_{window}"].append(
                _average(league_recent_wins[window][away_league_key])
            )
            form_values[f"home_league_recent_serve_pct_{window}"].append(
                _average(league_recent_serve_pct[window][home_league_key])
            )
            form_values[f"away_league_recent_serve_pct_{window}"].append(
                _average(league_recent_serve_pct[window][away_league_key])
            )
        home_days_since_last.append(_days_since(recent_dates[home_team], match_date))
        away_days_since_last.append(_days_since(recent_dates[away_team], match_date))

        home_win = 1 if row.winner == 1 else 0
        away_win = 1 if row.winner == 2 else 0
        home_weighted_win = home_win * (1.0 + home_opponent_class)
        away_weighted_win = away_win * (1.0 + away_opponent_class)

        # Update history only from resolved matches to avoid leaking live match state
        # into later rows in the same feature build.
        if row.winner in (1, 2):
            for window in FORM_WINDOWS:
                recent_wins[window][home_team].append(home_win)
                recent_wins[window][away_team].append(away_win)
                recent_opponent_classes[window][home_team].append(home_opponent_class)
                recent_opponent_classes[window][away_team].append(away_opponent_class)
                recent_weighted_wins[window][home_team].append(home_weighted_win)
                recent_weighted_wins[window][away_team].append(away_weighted_win)
                recent_strength_of_schedule[window][home_team].append(home_opponent_class)
                recent_strength_of_schedule[window][away_team].append(away_opponent_class)
                recent_serve_pct[window][home_team].append(home_match_serve_pct)
                recent_serve_pct[window][away_team].append(away_match_serve_pct)
                recent_serve_volume[window][home_team].append(home_match_serves)
                recent_serve_volume[window][away_team].append(away_match_serves)
                league_recent_wins[window][home_league_key].append(home_win)
                league_recent_wins[window][away_league_key].append(away_win)
                league_recent_serve_pct[window][home_league_key].append(home_match_serve_pct)
                league_recent_serve_pct[window][away_league_key].append(away_match_serve_pct)
        recent_dates[home_team] = match_date
        recent_dates[away_team] = match_date

    derived_values: dict[str, list[float]] = {}
    for window in FORM_WINDOWS:
        derived_values[f"recent_win_rate_gap_{window}"] = [
            home - away
            for home, away in zip(
                form_values[f"home_recent_win_rate_{window}"],
                form_values[f"away_recent_win_rate_{window}"],
            )
        ]
        derived_values[f"recent_opp_class_gap_{window}"] = [
            home - away
            for home, away in zip(
                form_values[f"home_recent_opp_class_avg_{window}"],
                form_values[f"away_recent_opp_class_avg_{window}"],
            )
        ]
        derived_values[f"recent_weighted_win_score_gap_{window}"] = [
            home - away
            for home, away in zip(
                form_values[f"home_recent_weighted_win_score_{window}"],
                form_values[f"away_recent_weighted_win_score_{window}"],
            )
        ]
        derived_values[f"recent_schedule_strength_gap_{window}"] = [
            home - away
            for home, away in zip(
                form_values[f"home_recent_schedule_strength_{window}"],
                form_values[f"away_recent_schedule_strength_{window}"],
            )
        ]
        derived_values[f"recent_serve_pct_gap_{window}"] = [
            home - away
            for home, away in zip(
                form_values[f"home_recent_serve_pct_{window}"],
                form_values[f"away_recent_serve_pct_{window}"],
            )
        ]
        derived_values[f"recent_serve_volume_gap_{window}"] = [
            home - away
            for home, away in zip(
                form_values[f"home_recent_serve_volume_{window}"],
                form_values[f"away_recent_serve_volume_{window}"],
            )
        ]
        derived_values[f"league_recent_win_rate_gap_{window}"] = [
            home - away
            for home, away in zip(
                form_values[f"home_league_recent_win_rate_{window}"],
                form_values[f"away_league_recent_win_rate_{window}"],
            )
        ]
        derived_values[f"league_recent_serve_pct_gap_{window}"] = [
            home - away
            for home, away in zip(
                form_values[f"home_league_recent_serve_pct_{window}"],
                form_values[f"away_league_recent_serve_pct_{window}"],
            )
        ]

    time_values = {
        "home_days_since_last": home_days_since_last,
        "away_days_since_last": away_days_since_last,
        "rest_days_gap": [
            home - away for home, away in zip(home_days_since_last, away_days_since_last)
        ],
    }
    features_block = pd.DataFrame({**form_values, **derived_values, **time_values}, index=df.index)
    return pd.concat([df, features_block], axis=1)


def get_feature_columns() -> list[str]:
    feature_columns = BASE_FEATURE_COLUMNS.copy()
    for window in FORM_WINDOWS:
        feature_columns.extend(
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
                f"home_recent_serve_pct_{window}",
                f"away_recent_serve_pct_{window}",
                f"recent_serve_pct_gap_{window}",
                f"home_recent_serve_volume_{window}",
                f"away_recent_serve_volume_{window}",
                f"recent_serve_volume_gap_{window}",
                f"home_league_recent_win_rate_{window}",
                f"away_league_recent_win_rate_{window}",
                f"league_recent_win_rate_gap_{window}",
                f"home_league_recent_serve_pct_{window}",
                f"away_league_recent_serve_pct_{window}",
                f"league_recent_serve_pct_gap_{window}",
            ]
        )
    return feature_columns


def prepare_feature_frame(matches: pd.DataFrame) -> pd.DataFrame:
    df = matches.copy()
    df["match_date"] = pd.to_datetime(df["match_date"])
    df = df.sort_values("match_date").reset_index(drop=True)
    df = add_form_features(df)
    df["winner"] = pd.to_numeric(df["winner"], errors="coerce")

    live_home_serve_pct = pd.to_numeric(df.get("live_home_serve_pct", 0.0), errors="coerce")
    live_away_serve_pct = pd.to_numeric(df.get("live_away_serve_pct", 0.0), errors="coerce")
    live_home_serve_volume = pd.to_numeric(
        df.get("live_home_serve_volume", 0.0), errors="coerce"
    )
    live_away_serve_volume = pd.to_numeric(
        df.get("live_away_serve_volume", 0.0), errors="coerce"
    )

    derived_block = pd.DataFrame(
        {
            "odds_gap": df["away_odds"] - df["home_odds"],
            "implied_home_prob": 1.0 / df["home_odds"],
            "implied_away_prob": 1.0 / df["away_odds"],
            "class_gap": df["team1_class"] - df["team2_class"],
            "is_women": (df["gender"] == "W").astype(int),
            "is_best_of_five": (df["best_of"] == 5).astype(int),
            "odds_source_opening": (df["odds_source"] == "opening").astype(int),
            "odds_source_first_seen": (df["odds_source"] == "first_seen").astype(int),
            "live_home_serve_pct": live_home_serve_pct,
            "live_away_serve_pct": live_away_serve_pct,
            "live_serve_pct_gap": live_home_serve_pct - live_away_serve_pct,
            "live_home_serve_volume": live_home_serve_volume,
            "live_away_serve_volume": live_away_serve_volume,
            "live_serve_volume_gap": live_home_serve_volume - live_away_serve_volume,
        },
        index=df.index,
    )
    df = pd.concat([df, derived_block], axis=1)

    numeric_columns = get_feature_columns()
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    return df


def build_features(matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = prepare_feature_frame(matches)
    x = df[get_feature_columns()].fillna(0.0)
    y = (df["winner"] == 1).astype(int)
    return x, y


def build_inference_features(matches: pd.DataFrame) -> pd.DataFrame:
    df = prepare_feature_frame(matches)
    return df[get_feature_columns()].fillna(0.0)
