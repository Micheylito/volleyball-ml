from __future__ import annotations

import pandas as pd

from src.config import settings
from src.db import load_live_matches, load_matches
from src.features import build_inference_features
from src.predict import predict_match
from src.telegram_bot import send_telegram_message


def format_live_message(row: pd.Series, probability: float) -> str:
    return (
        "Volleyball live signal\n"
        f"Match ID: {row['match_id']}\n"
        f"{row['home_team']} vs {row['away_team']}\n"
        f"League: {row['league']}\n"
        f"Status: {row['status']}\n"
        f"Home win probability: {probability:.2%}\n"
        f"Odds source: {row['odds_source']}\n"
        f"Live serve gap: {row['live_serve_pct_gap']:.4f}\n"
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
    for row_index, match in live_matches.reset_index(drop=True).iterrows():
        probability = predict_match(live_feature_frame.iloc[row_index].to_dict())
        print(
            f"match_id={match['match_id']} "
            f"{match['home_team']} vs {match['away_team']} "
            f"probability={probability:.4f}"
        )
        if probability >= settings.prediction_threshold:
            send_telegram_message(format_live_message(match, probability))
            sent_count += 1

    print(f"Processed live matches: {len(live_matches)}")
    print(f"Signals sent: {sent_count}")


if __name__ == "__main__":
    main()
