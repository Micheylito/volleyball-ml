from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine, text

from src.config import settings


DEFAULT_QUERY = """
SELECT
    m.id AS match_id,
    m.created_at AS match_date,
    m.tournament AS league,
    m.country,
    m.gender,
    m.age_group,
    m.best_of,
    m.team1 AS home_team,
    m.team2 AS away_team,
    m.status,
    m.winner,
    m.team1_class,
    m.team2_class,
    m.match_class,
    o.match_win1 AS home_odds,
    o.match_win2 AS away_odds,
    o.match_total_line,
    o.match_total_over,
    o.match_total_under,
    o.set1_win1,
    o.set1_win2,
    o.set1_total_line,
    o.set1_total_over,
    o.set1_total_under
FROM matches m
LEFT JOIN (
    SELECT mo.*
    FROM match_opening_odds mo
    INNER JOIN (
        SELECT match_id, MIN(ts) AS first_ts
        FROM match_opening_odds
        GROUP BY match_id
    ) first_odds
        ON mo.match_id = first_odds.match_id
       AND mo.ts = first_odds.first_ts
) o
    ON o.match_id = m.id
WHERE m.status = 'FINISHED'
  AND m.winner IN (1, 2)
  AND COALESCE(m.abandoned, 0) = 0
ORDER BY m.created_at ASC
"""


def load_matches(query: str = DEFAULT_QUERY) -> pd.DataFrame:
    if not settings.db_url:
        raise ValueError("DB_URL is empty. Fill .env before loading matches.")

    engine = create_engine(settings.db_url)
    with engine.connect() as connection:
        matches = pd.read_sql(text(query), connection)

    if matches.empty:
        raise ValueError("The query returned no finished matches. Check DB_URL and source tables.")

    return matches
