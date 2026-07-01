from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine, text

from src.config import settings


DEFAULT_QUERY = """
SELECT
    match_date,
    league,
    home_team,
    away_team,
    home_sets,
    away_sets,
    home_odds,
    away_odds
FROM volleyball_matches
WHERE home_sets IS NOT NULL
  AND away_sets IS NOT NULL
ORDER BY match_date ASC
"""


def load_matches(query: str = DEFAULT_QUERY) -> pd.DataFrame:
    if not settings.db_url:
        raise ValueError("DB_URL is empty. Fill .env before loading matches.")

    engine = create_engine(settings.db_url)
    with engine.connect() as connection:
        return pd.read_sql(text(query), connection)

