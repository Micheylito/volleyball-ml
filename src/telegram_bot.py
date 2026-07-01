from __future__ import annotations

import requests

from src.config import settings


def send_telegram_message(message: str) -> None:
    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        raise ValueError("Telegram settings are empty. Fill TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")

    url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
    response = requests.post(
        url,
        json={"chat_id": settings.telegram_chat_id, "text": message},
        timeout=30,
    )
    response.raise_for_status()

