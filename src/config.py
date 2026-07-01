from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    db_url: str = os.getenv("DB_URL", "")
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")
    model_path: str = os.getenv("MODEL_PATH", "models/match_model.joblib")
    data_export_path: str = os.getenv("DATA_EXPORT_PATH", "data/raw/matches.csv")
    prediction_threshold: float = float(os.getenv("PREDICTION_THRESHOLD", "0.62"))


settings = Settings()

