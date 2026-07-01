from __future__ import annotations

import pandas as pd

from src.config import settings
from src.db import load_live_matches, load_matches
from src.features import build_inference_features
from src.predict import predict_match
from src.telegram_bot import send_telegram_message


def _safe_text(value: object, fallback: str = "unknown") -> str:
    if pd.isna(value):
        return fallback
    return str(value)


def _safe_float(value: object, fallback: float = 0.0) -> float:
    if pd.isna(value):
        return fallback
    return float(value)


def format_live_message(row: pd.Series, probability: float) -> str:
    return (
        "Volleyball live signal\n"
        f"Match ID: {row['match_id']}\n"
        f"{_safe_text(row.get('home_team'))} vs {_safe_text(row.get('away_team'))}\n"
        f"League: {_safe_text(row.get('league'))}\n"
        f"Status: {_safe_text(row.get('status'))}\n"
        f"Home win probability: {probability:.2%}\n"
        f"Odds source: {_safe_text(row.get('odds_source'))}\n"
        f"Live serve gap: {_safe_float(row.get('live_serve_pct_gap')):.4f}\n"
        f"Threshold: {settings.prediction_threshold:.2%}"
    )


def main() -> None:
    historical_matches = load_matches()
    live_matches = load_live_matches()

    if live_matches.empty:
        print("No live matches with odds found.")
        return

    combined = pd.concat([historical_matches, live_matches], ignore_index=True)
    feature_frame = build_inference_features(combined)
    live_feature_frame = feature_frame.tail(len(live_matches)).reset_index(drop=True)

    sent_count = 0
    candidate_count = 0
    for row_index, match in live_matches.reset_index(drop=True).iterrows():
        live_features = live_feature_frame.iloc[row_index]
        probability = predict_match(live_features.to_dict())
        enriched_match = pd.concat([match, live_features])
        print(
            f"match_id={match['match_id']} "
            f"{_safe_text(match.get('home_team'))} vs {_safe_text(match.get('away_team'))} "
            f"probability={probability:.4f}"
        )
        if probability >= settings.prediction_threshold:
            candidate_count += 1
            if settings.send_telegram_signals:
                send_telegram_message(format_live_message(enriched_match, probability))
                sent_count += 1

    print(f"Processed live matches: {len(live_matches)}")
    print(f"Candidates above threshold: {candidate_count}")
    if not settings.send_telegram_signals:
        print("Telegram sending is disabled (SEND_TELEGRAM_SIGNALS=false).")
    print(f"Signals sent: {sent_count}")


if __name__ == "__main__":
    main()
