from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


DEFAULT_FEATURE_BLOCKS = (
    "core_market",
    "rest",
    "form_base",
    "serve_form",
    "league_form",
    "context_form",
    "live_serve",
)


def _parse_feature_blocks(raw_value: str) -> tuple[str, ...]:
    blocks = [block.strip() for block in raw_value.split(",") if block.strip()]
    return tuple(blocks) if blocks else DEFAULT_FEATURE_BLOCKS


@dataclass(frozen=True)
class Settings:
    db_url: str = os.getenv("DB_URL", "")
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")
    model_path: str = os.getenv("MODEL_PATH", "models/match_model.joblib")
    data_export_path: str = os.getenv("DATA_EXPORT_PATH", "data/raw/matches.csv")
    prediction_threshold: float = float(os.getenv("PREDICTION_THRESHOLD", "0.62"))
    feature_blocks: tuple[str, ...] = _parse_feature_blocks(
        os.getenv("FEATURE_BLOCKS", ",".join(DEFAULT_FEATURE_BLOCKS))
    )
    send_telegram_signals: bool = os.getenv("SEND_TELEGRAM_SIGNALS", "false").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


settings = Settings()
