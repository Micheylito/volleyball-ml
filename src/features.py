from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

import pandas as pd

from src.config import settings


FORM_WINDOWS = (3, 5, 10)
CORE_MARKET_COLUMNS = [
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
    "odds_source_opening",
    "odds_source_first_seen",
]
REST_COLUMNS = [
    "home_days_since_last",
    "away_days_since_last",
    "rest_days_gap",
]
LIVE_SERVE_COLUMNS = [
    "live_home_serve_pct",
    "live_away_serve_pct",
    "live_serve_pct_gap",
    "live_home_serve_volume",
    "live_away_serve_volume",
    "live_serve_volume_gap",
]
MARKET_DERIVED_COLUMNS = [
    "market_overround",
    "market_home_prob_norm",
    "market_away_prob_norm",
    "market_prob_gap_norm",
    "market_favorite_odds",
    "market_underdog_odds",
    "market_favorite_prob_norm",
    "market_underdog_prob_norm",
    "market_is_home_favorite",
    "market_is_away_favorite",
    "market_favorite_edge",
    "set1_overround",
    "set1_home_prob_norm",
    "set1_away_prob_norm",
    "set1_prob_gap_norm",
    "set1_home_favorite",
    "set1_away_favorite",
    "set1_vs_match_home_prob_delta",
    "set1_vs_match_away_prob_delta",
    "set1_vs_match_gap_delta",
]
MARKET_INTERACTION_COLUMNS = [
    "market_gap_ratio_set1_to_match",
    "market_abs_gap_ratio_set1_to_match",
    "market_favorite_flip_between_match_and_set1",
    "market_home_favorite_strength_x_set1_delta",
    "market_away_favorite_strength_x_set1_delta",
    "market_total_line_per_set",
    "market_total_line_vs_match_gap",
    "market_total_line_vs_favorite_edge",
    "market_set1_total_line_delta_vs_match_total",
]
LEAGUE_RELIABILITY_COLUMNS = [
    "league_reliability_match_count_3",
    "league_reliability_home_win_rate_3",
    "league_reliability_favorite_win_rate_3",
    "league_reliability_market_confidence_3",
    "league_reliability_market_error_3",
    "league_reliability_match_count_5",
    "league_reliability_home_win_rate_5",
    "league_reliability_favorite_win_rate_5",
    "league_reliability_market_confidence_5",
    "league_reliability_market_error_5",
    "league_reliability_match_count_10",
    "league_reliability_home_win_rate_10",
    "league_reliability_favorite_win_rate_10",
    "league_reliability_market_confidence_10",
    "league_reliability_market_error_10",
]
FEATURE_BLOCKS = (
    "core_market",
    "market_derived",
    "market_interactions",
    "league_reliability",
    "rest",
    "form_base",
    "serve_form",
    "league_form",
    "context_form",
    "live_serve",
    "set_trends",
)


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
    recent_set_win_rate = {
        window: defaultdict(lambda: deque(maxlen=window)) for window in FORM_WINDOWS
    }
    recent_match_length_ratio = {
        window: defaultdict(lambda: deque(maxlen=window)) for window in FORM_WINDOWS
    }
    recent_decider_rate = {
        window: defaultdict(lambda: deque(maxlen=window)) for window in FORM_WINDOWS
    }
    recent_sweep_win_rate = {
        window: defaultdict(lambda: deque(maxlen=window)) for window in FORM_WINDOWS
    }
    recent_swept_loss_rate = {
        window: defaultdict(lambda: deque(maxlen=window)) for window in FORM_WINDOWS
    }
    league_reliability_home_wins = {
        window: defaultdict(lambda: deque(maxlen=window)) for window in FORM_WINDOWS
    }
    league_reliability_favorite_wins = {
        window: defaultdict(lambda: deque(maxlen=window)) for window in FORM_WINDOWS
    }
    league_reliability_market_confidence = {
        window: defaultdict(lambda: deque(maxlen=window)) for window in FORM_WINDOWS
    }
    league_reliability_market_error = {
        window: defaultdict(lambda: deque(maxlen=window)) for window in FORM_WINDOWS
    }
    league_recent_wins = {
        window: defaultdict(lambda: deque(maxlen=window)) for window in FORM_WINDOWS
    }
    league_recent_serve_pct = {
        window: defaultdict(lambda: deque(maxlen=window)) for window in FORM_WINDOWS
    }
    context_recent_wins = {
        window: defaultdict(lambda: deque(maxlen=window)) for window in FORM_WINDOWS
    }
    context_recent_serve_pct = {
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
        form_values[f"home_recent_set_win_rate_{window}"] = []
        form_values[f"away_recent_set_win_rate_{window}"] = []
        form_values[f"home_recent_match_length_ratio_{window}"] = []
        form_values[f"away_recent_match_length_ratio_{window}"] = []
        form_values[f"home_recent_decider_rate_{window}"] = []
        form_values[f"away_recent_decider_rate_{window}"] = []
        form_values[f"home_recent_sweep_win_rate_{window}"] = []
        form_values[f"away_recent_sweep_win_rate_{window}"] = []
        form_values[f"home_recent_swept_loss_rate_{window}"] = []
        form_values[f"away_recent_swept_loss_rate_{window}"] = []
        form_values[f"home_league_recent_win_rate_{window}"] = []
        form_values[f"away_league_recent_win_rate_{window}"] = []
        form_values[f"home_league_recent_serve_pct_{window}"] = []
        form_values[f"away_league_recent_serve_pct_{window}"] = []
        form_values[f"home_context_recent_win_rate_{window}"] = []
        form_values[f"away_context_recent_win_rate_{window}"] = []
        form_values[f"home_context_recent_serve_pct_{window}"] = []
        form_values[f"away_context_recent_serve_pct_{window}"] = []
        form_values[f"league_reliability_match_count_{window}"] = []
        form_values[f"league_reliability_home_win_rate_{window}"] = []
        form_values[f"league_reliability_favorite_win_rate_{window}"] = []
        form_values[f"league_reliability_market_confidence_{window}"] = []
        form_values[f"league_reliability_market_error_{window}"] = []
    home_days_since_last: list[float] = []
    away_days_since_last: list[float] = []

    for row in df.itertuples(index=False):
        match_date = row.match_date
        home_team = row.home_team
        away_team = row.away_team
        league = row.league if pd.notna(row.league) else "unknown"
        country = row.country if pd.notna(row.country) else "unknown"
        gender = row.gender if pd.notna(row.gender) else "unknown"
        age_group = row.age_group if pd.notna(row.age_group) else "unknown"
        context = (country, gender, age_group)
        home_league_key = (home_team, league)
        away_league_key = (away_team, league)
        home_context_key = (home_team, context)
        away_context_key = (away_team, context)
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
        completed_sets = float(row.completed_sets) if pd.notna(row.completed_sets) else 0.0
        home_sets_won = float(row.home_sets_won) if pd.notna(row.home_sets_won) else 0.0
        away_sets_won = float(row.away_sets_won) if pd.notna(row.away_sets_won) else 0.0
        best_of_value = float(row.best_of) if pd.notna(row.best_of) and row.best_of else 0.0
        home_odds = float(row.home_odds) if pd.notna(row.home_odds) else 0.0
        away_odds = float(row.away_odds) if pd.notna(row.away_odds) else 0.0
        has_match_odds = home_odds > 0 and away_odds > 0
        implied_home_prob = (1.0 / home_odds) if has_match_odds else 0.0
        implied_away_prob = (1.0 / away_odds) if has_match_odds else 0.0
        market_overround = implied_home_prob + implied_away_prob
        market_home_prob_norm = (
            implied_home_prob / market_overround if has_match_odds and market_overround > 0 else 0.0
        )
        market_away_prob_norm = (
            implied_away_prob / market_overround if has_match_odds and market_overround > 0 else 0.0
        )
        market_prob_gap_norm = market_home_prob_norm - market_away_prob_norm

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
            form_values[f"home_recent_set_win_rate_{window}"].append(
                _average(recent_set_win_rate[window][home_team])
            )
            form_values[f"away_recent_set_win_rate_{window}"].append(
                _average(recent_set_win_rate[window][away_team])
            )
            form_values[f"home_recent_match_length_ratio_{window}"].append(
                _average(recent_match_length_ratio[window][home_team])
            )
            form_values[f"away_recent_match_length_ratio_{window}"].append(
                _average(recent_match_length_ratio[window][away_team])
            )
            form_values[f"home_recent_decider_rate_{window}"].append(
                _average(recent_decider_rate[window][home_team])
            )
            form_values[f"away_recent_decider_rate_{window}"].append(
                _average(recent_decider_rate[window][away_team])
            )
            form_values[f"home_recent_sweep_win_rate_{window}"].append(
                _average(recent_sweep_win_rate[window][home_team])
            )
            form_values[f"away_recent_sweep_win_rate_{window}"].append(
                _average(recent_sweep_win_rate[window][away_team])
            )
            form_values[f"home_recent_swept_loss_rate_{window}"].append(
                _average(recent_swept_loss_rate[window][home_team])
            )
            form_values[f"away_recent_swept_loss_rate_{window}"].append(
                _average(recent_swept_loss_rate[window][away_team])
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
            form_values[f"home_context_recent_win_rate_{window}"].append(
                _average(context_recent_wins[window][home_context_key])
            )
            form_values[f"away_context_recent_win_rate_{window}"].append(
                _average(context_recent_wins[window][away_context_key])
            )
            form_values[f"home_context_recent_serve_pct_{window}"].append(
                _average(context_recent_serve_pct[window][home_context_key])
            )
            form_values[f"away_context_recent_serve_pct_{window}"].append(
                _average(context_recent_serve_pct[window][away_context_key])
            )
            form_values[f"league_reliability_match_count_{window}"].append(
                len(league_reliability_home_wins[window][league])
            )
            form_values[f"league_reliability_home_win_rate_{window}"].append(
                _average(league_reliability_home_wins[window][league])
            )
            form_values[f"league_reliability_favorite_win_rate_{window}"].append(
                _average(league_reliability_favorite_wins[window][league])
            )
            form_values[f"league_reliability_market_confidence_{window}"].append(
                _average(league_reliability_market_confidence[window][league])
            )
            form_values[f"league_reliability_market_error_{window}"].append(
                _average(league_reliability_market_error[window][league])
            )
        home_days_since_last.append(_days_since(recent_dates[home_team], match_date))
        away_days_since_last.append(_days_since(recent_dates[away_team], match_date))

        home_win = 1 if row.winner == 1 else 0
        away_win = 1 if row.winner == 2 else 0
        home_weighted_win = home_win * (1.0 + home_opponent_class)
        away_weighted_win = away_win * (1.0 + away_opponent_class)
        total_sets = completed_sets if completed_sets > 0 else home_sets_won + away_sets_won
        home_set_win_rate = home_sets_won / total_sets if total_sets > 0 else 0.0
        away_set_win_rate = away_sets_won / total_sets if total_sets > 0 else 0.0
        match_length_ratio = total_sets / best_of_value if best_of_value > 0 else 0.0
        decider_played = 1.0 if best_of_value > 0 and total_sets >= best_of_value else 0.0
        home_sweep_win = 1.0 if home_win == 1 and away_sets_won == 0 and total_sets > 0 else 0.0
        away_sweep_win = 1.0 if away_win == 1 and home_sets_won == 0 and total_sets > 0 else 0.0
        home_swept_loss = 1.0 if home_win == 0 and home_sets_won == 0 and total_sets > 0 else 0.0
        away_swept_loss = 1.0 if away_win == 0 and away_sets_won == 0 and total_sets > 0 else 0.0
        favorite_win = 0.0
        market_confidence = abs(market_prob_gap_norm)
        market_error = abs(home_win - market_home_prob_norm) if has_match_odds else 0.0
        if has_match_odds:
            if home_odds < away_odds:
                favorite_win = float(home_win)
            elif away_odds < home_odds:
                favorite_win = float(away_win)
            else:
                favorite_win = 0.5

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
                recent_set_win_rate[window][home_team].append(home_set_win_rate)
                recent_set_win_rate[window][away_team].append(away_set_win_rate)
                recent_match_length_ratio[window][home_team].append(match_length_ratio)
                recent_match_length_ratio[window][away_team].append(match_length_ratio)
                recent_decider_rate[window][home_team].append(decider_played)
                recent_decider_rate[window][away_team].append(decider_played)
                recent_sweep_win_rate[window][home_team].append(home_sweep_win)
                recent_sweep_win_rate[window][away_team].append(away_sweep_win)
                recent_swept_loss_rate[window][home_team].append(home_swept_loss)
                recent_swept_loss_rate[window][away_team].append(away_swept_loss)
                if has_match_odds:
                    league_reliability_home_wins[window][league].append(float(home_win))
                    league_reliability_favorite_wins[window][league].append(favorite_win)
                    league_reliability_market_confidence[window][league].append(market_confidence)
                    league_reliability_market_error[window][league].append(market_error)
                league_recent_wins[window][home_league_key].append(home_win)
                league_recent_wins[window][away_league_key].append(away_win)
                league_recent_serve_pct[window][home_league_key].append(home_match_serve_pct)
                league_recent_serve_pct[window][away_league_key].append(away_match_serve_pct)
                context_recent_wins[window][home_context_key].append(home_win)
                context_recent_wins[window][away_context_key].append(away_win)
                context_recent_serve_pct[window][home_context_key].append(home_match_serve_pct)
                context_recent_serve_pct[window][away_context_key].append(away_match_serve_pct)
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
        derived_values[f"recent_set_win_rate_gap_{window}"] = [
            home - away
            for home, away in zip(
                form_values[f"home_recent_set_win_rate_{window}"],
                form_values[f"away_recent_set_win_rate_{window}"],
            )
        ]
        derived_values[f"recent_match_length_ratio_gap_{window}"] = [
            home - away
            for home, away in zip(
                form_values[f"home_recent_match_length_ratio_{window}"],
                form_values[f"away_recent_match_length_ratio_{window}"],
            )
        ]
        derived_values[f"recent_decider_rate_gap_{window}"] = [
            home - away
            for home, away in zip(
                form_values[f"home_recent_decider_rate_{window}"],
                form_values[f"away_recent_decider_rate_{window}"],
            )
        ]
        derived_values[f"recent_sweep_win_rate_gap_{window}"] = [
            home - away
            for home, away in zip(
                form_values[f"home_recent_sweep_win_rate_{window}"],
                form_values[f"away_recent_sweep_win_rate_{window}"],
            )
        ]
        derived_values[f"recent_swept_loss_rate_gap_{window}"] = [
            home - away
            for home, away in zip(
                form_values[f"home_recent_swept_loss_rate_{window}"],
                form_values[f"away_recent_swept_loss_rate_{window}"],
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
        derived_values[f"context_recent_win_rate_gap_{window}"] = [
            home - away
            for home, away in zip(
                form_values[f"home_context_recent_win_rate_{window}"],
                form_values[f"away_context_recent_win_rate_{window}"],
            )
        ]
        derived_values[f"context_recent_serve_pct_gap_{window}"] = [
            home - away
            for home, away in zip(
                form_values[f"home_context_recent_serve_pct_{window}"],
                form_values[f"away_context_recent_serve_pct_{window}"],
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


def get_feature_columns(active_blocks: tuple[str, ...] | None = None) -> list[str]:
    selected_blocks = active_blocks or settings.feature_blocks
    unknown_blocks = [block for block in selected_blocks if block not in FEATURE_BLOCKS]
    if unknown_blocks:
        raise ValueError(
            f"Unknown feature blocks: {', '.join(unknown_blocks)}. "
            f"Allowed blocks: {', '.join(FEATURE_BLOCKS)}"
        )

    feature_columns: list[str] = []
    if "core_market" in selected_blocks:
        feature_columns.extend(CORE_MARKET_COLUMNS)
    if "rest" in selected_blocks:
        feature_columns.extend(REST_COLUMNS)
    if "live_serve" in selected_blocks:
        feature_columns.extend(LIVE_SERVE_COLUMNS)
    if "market_derived" in selected_blocks:
        feature_columns.extend(MARKET_DERIVED_COLUMNS)
    if "market_interactions" in selected_blocks:
        feature_columns.extend(MARKET_INTERACTION_COLUMNS)
    if "league_reliability" in selected_blocks:
        feature_columns.extend(LEAGUE_RELIABILITY_COLUMNS)

    for window in FORM_WINDOWS:
        if "form_base" in selected_blocks:
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
                ]
            )
        if "serve_form" in selected_blocks:
            feature_columns.extend(
                [
                    f"home_recent_serve_pct_{window}",
                    f"away_recent_serve_pct_{window}",
                    f"recent_serve_pct_gap_{window}",
                    f"home_recent_serve_volume_{window}",
                    f"away_recent_serve_volume_{window}",
                    f"recent_serve_volume_gap_{window}",
                ]
            )
        if "set_trends" in selected_blocks:
            feature_columns.extend(
                [
                    f"home_recent_set_win_rate_{window}",
                    f"away_recent_set_win_rate_{window}",
                    f"recent_set_win_rate_gap_{window}",
                    f"home_recent_match_length_ratio_{window}",
                    f"away_recent_match_length_ratio_{window}",
                    f"recent_match_length_ratio_gap_{window}",
                    f"home_recent_decider_rate_{window}",
                    f"away_recent_decider_rate_{window}",
                    f"recent_decider_rate_gap_{window}",
                    f"home_recent_sweep_win_rate_{window}",
                    f"away_recent_sweep_win_rate_{window}",
                    f"recent_sweep_win_rate_gap_{window}",
                    f"home_recent_swept_loss_rate_{window}",
                    f"away_recent_swept_loss_rate_{window}",
                    f"recent_swept_loss_rate_gap_{window}",
                ]
            )
        if "league_form" in selected_blocks:
            feature_columns.extend(
                [
                    f"home_league_recent_win_rate_{window}",
                    f"away_league_recent_win_rate_{window}",
                    f"league_recent_win_rate_gap_{window}",
                    f"home_league_recent_serve_pct_{window}",
                    f"away_league_recent_serve_pct_{window}",
                    f"league_recent_serve_pct_gap_{window}",
                ]
            )
        if "context_form" in selected_blocks:
            feature_columns.extend(
                [
                    f"home_context_recent_win_rate_{window}",
                    f"away_context_recent_win_rate_{window}",
                    f"context_recent_win_rate_gap_{window}",
                    f"home_context_recent_serve_pct_{window}",
                    f"away_context_recent_serve_pct_{window}",
                    f"context_recent_serve_pct_gap_{window}",
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
    home_odds = pd.to_numeric(df["home_odds"], errors="coerce")
    away_odds = pd.to_numeric(df["away_odds"], errors="coerce")
    set1_home_odds = pd.to_numeric(df["set1_win1"], errors="coerce")
    set1_away_odds = pd.to_numeric(df["set1_win2"], errors="coerce")

    implied_home_prob = 1.0 / home_odds
    implied_away_prob = 1.0 / away_odds
    market_overround = implied_home_prob + implied_away_prob
    market_home_prob_norm = implied_home_prob / market_overround
    market_away_prob_norm = implied_away_prob / market_overround
    market_prob_gap_norm = market_home_prob_norm - market_away_prob_norm

    market_favorite_odds = pd.concat([home_odds, away_odds], axis=1).min(axis=1)
    market_underdog_odds = pd.concat([home_odds, away_odds], axis=1).max(axis=1)
    market_favorite_prob_norm = pd.concat(
        [market_home_prob_norm, market_away_prob_norm], axis=1
    ).max(axis=1)
    market_underdog_prob_norm = pd.concat(
        [market_home_prob_norm, market_away_prob_norm], axis=1
    ).min(axis=1)
    market_is_home_favorite = (home_odds < away_odds).astype(int)
    market_is_away_favorite = (away_odds < home_odds).astype(int)
    market_favorite_edge = market_favorite_prob_norm - market_underdog_prob_norm

    set1_implied_home_prob = 1.0 / set1_home_odds
    set1_implied_away_prob = 1.0 / set1_away_odds
    set1_overround = set1_implied_home_prob + set1_implied_away_prob
    set1_home_prob_norm = set1_implied_home_prob / set1_overround
    set1_away_prob_norm = set1_implied_away_prob / set1_overround
    set1_prob_gap_norm = set1_home_prob_norm - set1_away_prob_norm
    set1_home_favorite = (set1_home_odds < set1_away_odds).astype(int)
    set1_away_favorite = (set1_away_odds < set1_home_odds).astype(int)
    match_total_line = pd.to_numeric(df["match_total_line"], errors="coerce")
    set1_total_line = pd.to_numeric(df["set1_total_line"], errors="coerce")
    best_of = pd.to_numeric(df["best_of"], errors="coerce")

    market_gap_ratio_set1_to_match = set1_prob_gap_norm / market_prob_gap_norm
    market_abs_gap_ratio_set1_to_match = set1_prob_gap_norm.abs() / market_prob_gap_norm.abs()
    market_favorite_flip_between_match_and_set1 = (
        market_is_home_favorite != set1_home_favorite
    ).astype(int)
    market_home_favorite_strength_x_set1_delta = (
        market_is_home_favorite * (set1_home_prob_norm - market_home_prob_norm)
    )
    market_away_favorite_strength_x_set1_delta = (
        market_is_away_favorite * (set1_away_prob_norm - market_away_prob_norm)
    )
    market_total_line_per_set = match_total_line / best_of
    market_total_line_vs_match_gap = match_total_line / market_prob_gap_norm.abs()
    market_total_line_vs_favorite_edge = match_total_line / market_favorite_edge
    market_set1_total_line_delta_vs_match_total = set1_total_line - market_total_line_per_set

    derived_block = pd.DataFrame(
        {
            "odds_gap": away_odds - home_odds,
            "implied_home_prob": implied_home_prob,
            "implied_away_prob": implied_away_prob,
            "class_gap": df["team1_class"] - df["team2_class"],
            "is_women": (df["gender"] == "W").astype(int),
            "is_best_of_five": (df["best_of"] == 5).astype(int),
            "odds_source_opening": (df["odds_source"] == "opening").astype(int),
            "odds_source_first_seen": (df["odds_source"] == "first_seen").astype(int),
            "market_overround": market_overround,
            "market_home_prob_norm": market_home_prob_norm,
            "market_away_prob_norm": market_away_prob_norm,
            "market_prob_gap_norm": market_prob_gap_norm,
            "market_favorite_odds": market_favorite_odds,
            "market_underdog_odds": market_underdog_odds,
            "market_favorite_prob_norm": market_favorite_prob_norm,
            "market_underdog_prob_norm": market_underdog_prob_norm,
            "market_is_home_favorite": market_is_home_favorite,
            "market_is_away_favorite": market_is_away_favorite,
            "market_favorite_edge": market_favorite_edge,
            "set1_overround": set1_overround,
            "set1_home_prob_norm": set1_home_prob_norm,
            "set1_away_prob_norm": set1_away_prob_norm,
            "set1_prob_gap_norm": set1_prob_gap_norm,
            "set1_home_favorite": set1_home_favorite,
            "set1_away_favorite": set1_away_favorite,
            "set1_vs_match_home_prob_delta": set1_home_prob_norm - market_home_prob_norm,
            "set1_vs_match_away_prob_delta": set1_away_prob_norm - market_away_prob_norm,
            "set1_vs_match_gap_delta": set1_prob_gap_norm - market_prob_gap_norm,
            "market_gap_ratio_set1_to_match": market_gap_ratio_set1_to_match,
            "market_abs_gap_ratio_set1_to_match": market_abs_gap_ratio_set1_to_match,
            "market_favorite_flip_between_match_and_set1": (
                market_favorite_flip_between_match_and_set1
            ),
            "market_home_favorite_strength_x_set1_delta": (
                market_home_favorite_strength_x_set1_delta
            ),
            "market_away_favorite_strength_x_set1_delta": (
                market_away_favorite_strength_x_set1_delta
            ),
            "market_total_line_per_set": market_total_line_per_set,
            "market_total_line_vs_match_gap": market_total_line_vs_match_gap,
            "market_total_line_vs_favorite_edge": market_total_line_vs_favorite_edge,
            "market_set1_total_line_delta_vs_match_total": (
                market_set1_total_line_delta_vs_match_total
            ),
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
    df = df.replace([np.inf, -np.inf], np.nan)

    numeric_columns = list(dict.fromkeys(get_feature_columns()))
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    return df


def build_features(
    matches: pd.DataFrame, active_blocks: tuple[str, ...] | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    df = prepare_feature_frame(matches)
    selected_blocks = active_blocks or settings.feature_blocks
    x = df[get_feature_columns(selected_blocks)].fillna(0.0)
    y = (df["winner"] == 1).astype(int)
    return x, y


def build_inference_features(
    matches: pd.DataFrame, active_blocks: tuple[str, ...] | None = None
) -> pd.DataFrame:
    df = prepare_feature_frame(matches)
    selected_blocks = active_blocks or settings.feature_blocks
    return df[get_feature_columns(selected_blocks)].fillna(0.0)
