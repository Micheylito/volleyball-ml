from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine, text

from src.config import settings


CURRENT_SET_LIVE_QUERY = """
WITH latest_rally_odds AS (
    SELECT *
    FROM (
        SELECT
            ro.*,
            ROW_NUMBER() OVER (
                PARTITION BY ro.rally_db_id
                ORDER BY ro.ts DESC, ro.id DESC
            ) AS rn
        FROM rally_odds ro
        WHERE ro.rally_db_id IS NOT NULL
          AND ro.odds_status = 'OK'
    ) ranked
    WHERE rn = 1
),
set_results AS (
    SELECT
        s.match_id,
        s.set_number,
        s.winner AS set_winner,
        s.score1 AS final_score1,
        s.score2 AS final_score2,
        s.created_at AS set_finished_at
    FROM sets s
    WHERE COALESCE(s.finished, 0) = 1
      AND s.winner IN (1, 2)
),
match_context AS (
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
        m.team1_class,
        m.team2_class,
        m.match_class
    FROM matches m
    WHERE COALESCE(m.abandoned, 0) = 0
)
SELECT
    mc.match_id,
    COALESCE(lo.ts, r.created_at) AS snapshot_ts,
    mc.match_date,
    mc.league,
    mc.country,
    mc.gender,
    mc.age_group,
    mc.best_of,
    mc.home_team,
    mc.away_team,
    mc.team1_class,
    mc.team2_class,
    mc.match_class,
    r.id AS rally_db_id,
    r.set_number,
    r.rally_number,
    r.score1,
    r.score2,
    r.serve_team,
    lo.set_win1,
    lo.set_win2,
    lo.set_total_line,
    lo.set_total_over,
    lo.set_total_under,
    lo.match_win1,
    lo.match_win2,
    sr.set_winner,
    sr.final_score1,
    sr.final_score2,
    CASE WHEN sr.set_winner = 1 THEN 1 ELSE 0 END AS target_set_team1_win
FROM rallies r
INNER JOIN latest_rally_odds lo
    ON lo.rally_db_id = r.id
INNER JOIN set_results sr
    ON sr.match_id = r.match_id
   AND sr.set_number = r.set_number
INNER JOIN match_context mc
    ON mc.match_id = r.match_id
WHERE lo.set_win1 IS NOT NULL
  AND lo.set_win2 IS NOT NULL
ORDER BY COALESCE(lo.ts, r.created_at) ASC, mc.match_id ASC, r.set_number ASC, r.id ASC
"""


def load_current_set_live_rows(query: str = CURRENT_SET_LIVE_QUERY) -> pd.DataFrame:
    if not settings.db_url:
        raise ValueError("DB_URL is empty. Fill .env before loading current set live rows.")

    engine = create_engine(settings.db_url)
    with engine.connect() as connection:
        rows = pd.read_sql(text(query), connection)

    if rows.empty:
        raise ValueError("The current set live query returned no rows.")

    return rows
