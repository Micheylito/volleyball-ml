from __future__ import annotations

from src.config import settings
from src.predict import predict_match
from src.telegram_bot import send_telegram_message


def build_demo_match() -> dict[str, float]:
    # Placeholder values until we connect the real upcoming-match pipeline.
    return {
        "home_odds": 1.72,
        "away_odds": 2.15,
        "odds_gap": 0.43,
        "total_sets": 4.1,
    }


def main() -> None:
    match_features = build_demo_match()
    probability = predict_match(match_features)

    if probability >= settings.prediction_threshold:
        message = (
            "Volleyball signal\n"
            f"Home win probability: {probability:.2%}\n"
            f"Threshold: {settings.prediction_threshold:.2%}"
        )
        send_telegram_message(message)
        print("Signal sent to Telegram.")
    else:
        print(
            f"No signal. Probability {probability:.2%} is below threshold "
            f"{settings.prediction_threshold:.2%}."
        )


if __name__ == "__main__":
    main()

