from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine, text

from src.config import settings


DEFAULT_QUERY = """
WITH opening_odds AS (
    SELECT mo.*
    FROM match_opening_odds mo
    INNER JOIN (
        SELECT match_id, MIN(ts) AS first_ts
        FROM match_opening_odds
        GROUP BY match_id
    ) first_odds
        ON mo.match_id = first_odds.match_id
       AND mo.ts = first_odds.first_ts
),
first_seen_odds AS (
    SELECT fo.*
    FROM match_first_seen_odds fo
    INNER JOIN (
        SELECT match_id, MIN(ts) AS first_ts
        FROM match_first_seen_odds
        GROUP BY match_id
    ) first_seen
        ON fo.match_id = first_seen.match_id
       AND fo.ts = first_seen.first_ts
),
set_summary AS (
    SELECT
        s.match_id,
        SUM(CASE WHEN COALESCE(s.finished, 0) = 1 THEN 1 ELSE 0 END) AS completed_sets,
        SUM(CASE WHEN s.winner = 1 THEN 1 ELSE 0 END) AS home_sets_won,
        SUM(CASE WHEN s.winner = 2 THEN 1 ELSE 0 END) AS away_sets_won
    FROM sets s
    GROUP BY s.match_id
)
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
    COALESCE(o.match_win1, f.match_win1) AS home_odds,
    COALESCE(o.match_win2, f.match_win2) AS away_odds,
    COALESCE(o.match_total_line, f.match_total_line) AS match_total_line,
    COALESCE(o.match_total_over, f.match_total_over) AS match_total_over,
    COALESCE(o.match_total_under, f.match_total_under) AS match_total_under,
    COALESCE(o.set1_win1, f.set1_win1) AS set1_win1,
    COALESCE(o.set1_win2, f.set1_win2) AS set1_win2,
    COALESCE(o.set1_total_line, f.set1_total_line) AS set1_total_line,
    COALESCE(o.set1_total_over, f.set1_total_over) AS set1_total_over,
    COALESCE(o.set1_total_under, f.set1_total_under) AS set1_total_under,
    ms1.serves AS home_match_serves,
    ms1.success AS home_match_serve_success,
    COALESCE(ms1.pct, ms1.percent / 100.0) AS home_match_serve_pct,
    ms2.serves AS away_match_serves,
    ms2.success AS away_match_serve_success,
    COALESCE(ms2.pct, ms2.percent / 100.0) AS away_match_serve_pct,
    ss.completed_sets,
    ss.home_sets_won,
    ss.away_sets_won,
    CASE
        WHEN o.match_win1 IS NOT NULL AND o.match_win2 IS NOT NULL THEN 'opening'
        WHEN f.match_win1 IS NOT NULL AND f.match_win2 IS NOT NULL THEN 'first_seen'
        ELSE 'missing'
    END AS odds_source
FROM matches m
LEFT JOIN opening_odds o
    ON o.match_id = m.id
LEFT JOIN first_seen_odds f
    ON f.match_id = m.id
LEFT JOIN match_serve_summary ms1
    ON ms1.match_id = m.id
   AND ms1.team = 1
LEFT JOIN match_serve_summary ms2
    ON ms2.match_id = m.id
   AND ms2.team = 2
LEFT JOIN set_summary ss
    ON ss.match_id = m.id
WHERE m.status = 'FINISHED'
  AND m.winner IN (1, 2)
  AND COALESCE(m.abandoned, 0) = 0
ORDER BY m.created_at ASC
"""

DATASET_DIAGNOSTICS_QUERY = """
WITH opening_odds AS (
    SELECT mo.*
    FROM match_opening_odds mo
    INNER JOIN (
        SELECT match_id, MIN(ts) AS first_ts
        FROM match_opening_odds
        GROUP BY match_id
    ) first_odds
        ON mo.match_id = first_odds.match_id
       AND mo.ts = first_odds.first_ts
),
first_seen_odds AS (
    SELECT fo.*
    FROM match_first_seen_odds fo
    INNER JOIN (
        SELECT match_id, MIN(ts) AS first_ts
        FROM match_first_seen_odds
        GROUP BY match_id
    ) first_seen
        ON fo.match_id = first_seen.match_id
       AND fo.ts = first_seen.first_ts
),
set_summary AS (
    SELECT
        s.match_id,
        SUM(CASE WHEN COALESCE(s.finished, 0) = 1 THEN 1 ELSE 0 END) AS completed_sets,
        SUM(CASE WHEN s.winner = 1 THEN 1 ELSE 0 END) AS home_sets_won,
        SUM(CASE WHEN s.winner = 2 THEN 1 ELSE 0 END) AS away_sets_won
    FROM sets s
    GROUP BY s.match_id
)
SELECT 'all_matches' AS metric, COUNT(*) AS value
FROM matches
UNION ALL
SELECT 'finished_status', COUNT(*)
FROM matches
WHERE status = 'FINISHED'
UNION ALL
SELECT 'valid_winner', COUNT(*)
FROM matches
WHERE winner IN (1, 2)
UNION ALL
SELECT 'not_abandoned', COUNT(*)
FROM matches
WHERE COALESCE(abandoned, 0) = 0
UNION ALL
SELECT 'trainable_matches', COUNT(*)
FROM matches
WHERE status = 'FINISHED'
  AND winner IN (1, 2)
  AND COALESCE(abandoned, 0) = 0
UNION ALL
SELECT 'trainable_with_opening_odds', COUNT(*)
FROM matches m
LEFT JOIN opening_odds o
    ON o.match_id = m.id
WHERE m.status = 'FINISHED'
  AND m.winner IN (1, 2)
  AND COALESCE(m.abandoned, 0) = 0
  AND o.match_win1 IS NOT NULL
  AND o.match_win2 IS NOT NULL
UNION ALL
SELECT 'trainable_with_first_seen_odds', COUNT(*)
FROM matches m
LEFT JOIN first_seen_odds f
    ON f.match_id = m.id
WHERE m.status = 'FINISHED'
  AND m.winner IN (1, 2)
  AND COALESCE(m.abandoned, 0) = 0
  AND f.match_win1 IS NOT NULL
  AND f.match_win2 IS NOT NULL
UNION ALL
SELECT 'trainable_with_any_odds', COUNT(*)
FROM matches m
LEFT JOIN opening_odds o
    ON o.match_id = m.id
LEFT JOIN first_seen_odds f
    ON f.match_id = m.id
WHERE m.status = 'FINISHED'
  AND m.winner IN (1, 2)
  AND COALESCE(m.abandoned, 0) = 0
  AND COALESCE(o.match_win1, f.match_win1) IS NOT NULL
  AND COALESCE(o.match_win2, f.match_win2) IS NOT NULL
UNION ALL
SELECT 'trainable_with_match_serve_summary', COUNT(*)
FROM matches m
LEFT JOIN match_serve_summary ms1
    ON ms1.match_id = m.id
   AND ms1.team = 1
LEFT JOIN match_serve_summary ms2
    ON ms2.match_id = m.id
   AND ms2.team = 2
WHERE m.status = 'FINISHED'
  AND m.winner IN (1, 2)
  AND COALESCE(m.abandoned, 0) = 0
  AND COALESCE(ms1.pct, ms1.percent / 100.0) IS NOT NULL
  AND COALESCE(ms2.pct, ms2.percent / 100.0) IS NOT NULL
UNION ALL
SELECT 'trainable_with_completed_sets', COUNT(*)
FROM matches m
LEFT JOIN set_summary ss
    ON ss.match_id = m.id
WHERE m.status = 'FINISHED'
  AND m.winner IN (1, 2)
  AND COALESCE(m.abandoned, 0) = 0
  AND ss.completed_sets IS NOT NULL
"""

LIVE_MATCHES_QUERY = """
WITH opening_odds AS (
    SELECT mo.*
    FROM match_opening_odds mo
    INNER JOIN (
        SELECT match_id, MIN(ts) AS first_ts
        FROM match_opening_odds
        GROUP BY match_id
    ) first_odds
        ON mo.match_id = first_odds.match_id
       AND mo.ts = first_odds.first_ts
),
first_seen_odds AS (
    SELECT fo.*
    FROM match_first_seen_odds fo
    INNER JOIN (
        SELECT match_id, MIN(ts) AS first_ts
        FROM match_first_seen_odds
        GROUP BY match_id
    ) first_seen
        ON fo.match_id = first_seen.match_id
       AND fo.ts = first_seen.first_ts
),
live_serve AS (
    SELECT
        match_id,
        team,
        SUM(serves) AS serves,
        SUM(serve_wins) AS serve_wins,
        CASE
            WHEN SUM(serves) > 0 THEN CAST(SUM(serve_wins) AS REAL) / SUM(serves)
            ELSE NULL
        END AS pct
    FROM match_serve_set_stats
    GROUP BY match_id, team
)
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
    CAST(NULL AS INTEGER) AS winner,
    m.team1_class,
    m.team2_class,
    m.match_class,
    COALESCE(o.match_win1, f.match_win1) AS home_odds,
    COALESCE(o.match_win2, f.match_win2) AS away_odds,
    COALESCE(o.match_total_line, f.match_total_line) AS match_total_line,
    COALESCE(o.match_total_over, f.match_total_over) AS match_total_over,
    COALESCE(o.match_total_under, f.match_total_under) AS match_total_under,
    COALESCE(o.set1_win1, f.set1_win1) AS set1_win1,
    COALESCE(o.set1_win2, f.set1_win2) AS set1_win2,
    COALESCE(o.set1_total_line, f.set1_total_line) AS set1_total_line,
    COALESCE(o.set1_total_over, f.set1_total_over) AS set1_total_over,
    COALESCE(o.set1_total_under, f.set1_total_under) AS set1_total_under,
    CAST(NULL AS REAL) AS home_match_serves,
    CAST(NULL AS REAL) AS home_match_serve_success,
    CAST(NULL AS REAL) AS home_match_serve_pct,
    CAST(NULL AS REAL) AS away_match_serves,
    CAST(NULL AS REAL) AS away_match_serve_success,
    CAST(NULL AS REAL) AS away_match_serve_pct,
    CAST(NULL AS REAL) AS completed_sets,
    CAST(NULL AS REAL) AS home_sets_won,
    CAST(NULL AS REAL) AS away_sets_won,
    ls1.pct AS live_home_serve_pct,
    ls2.pct AS live_away_serve_pct,
    ls1.serves AS live_home_serve_volume,
    ls2.serves AS live_away_serve_volume,
    CASE
        WHEN o.match_win1 IS NOT NULL AND o.match_win2 IS NOT NULL THEN 'opening'
        WHEN f.match_win1 IS NOT NULL AND f.match_win2 IS NOT NULL THEN 'first_seen'
        ELSE 'missing'
    END AS odds_source
FROM matches m
LEFT JOIN opening_odds o
    ON o.match_id = m.id
LEFT JOIN first_seen_odds f
    ON f.match_id = m.id
LEFT JOIN live_serve ls1
    ON ls1.match_id = m.id
   AND ls1.team = 1
LEFT JOIN live_serve ls2
    ON ls2.match_id = m.id
   AND ls2.team = 2
WHERE m.status != 'FINISHED'
  AND COALESCE(m.abandoned, 0) = 0
  AND COALESCE(o.match_win1, f.match_win1) IS NOT NULL
  AND COALESCE(o.match_win2, f.match_win2) IS NOT NULL
ORDER BY m.created_at ASC
"""


def build_rally_backtest_query(match_ids: list[int]) -> str:
    unique_ids = sorted({int(match_id) for match_id in match_ids})
    if not unique_ids:
        raise ValueError("match_ids is empty.")

    ids_sql = ", ".join(str(match_id) for match_id in unique_ids)
    return f"""
WITH opening_odds AS (
    SELECT mo.*
    FROM match_opening_odds mo
    INNER JOIN (
        SELECT match_id, MIN(ts) AS first_ts
        FROM match_opening_odds
        GROUP BY match_id
    ) first_odds
        ON mo.match_id = first_odds.match_id
       AND mo.ts = first_odds.first_ts
),
first_seen_odds AS (
    SELECT fo.*
    FROM match_first_seen_odds fo
    INNER JOIN (
        SELECT match_id, MIN(ts) AS first_ts
        FROM match_first_seen_odds
        GROUP BY match_id
    ) first_seen
        ON fo.match_id = first_seen.match_id
       AND fo.ts = first_seen.first_ts
),

rallies_enriched AS (
    SELECT
        r.id AS rally_db_id,
        r.match_id,
        r.set_number,
        r.rally_number,
        r.score1,
        r.score2,
        r.created_at,
        r.serve_team,
        SUM(CASE WHEN r.serve_team = 1 THEN 1 ELSE 0 END) OVER (
            PARTITION BY r.match_id
            ORDER BY r.id
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS live_home_serve_volume,
        SUM(CASE WHEN r.serve_team = 2 THEN 1 ELSE 0 END) OVER (
            PARTITION BY r.match_id
            ORDER BY r.id
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS live_away_serve_volume,
        SUM(CASE WHEN r.serve_team = 1 AND r.point_winner = 1 THEN 1 ELSE 0 END) OVER (
            PARTITION BY r.match_id
            ORDER BY r.id
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS live_home_serve_wins,
        SUM(CASE WHEN r.serve_team = 2 AND r.point_winner = 2 THEN 1 ELSE 0 END) OVER (
            PARTITION BY r.match_id
            ORDER BY r.id
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS live_away_serve_wins
    FROM rallies r
    WHERE r.match_id IN ({ids_sql})
),
latest_rally_odds AS (
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
)
SELECT
    m.id AS match_id,
    COALESCE(lo.ts, re.created_at) AS match_date,
    m.tournament AS league,
    m.country,
    m.gender,
    m.age_group,
    m.best_of,
    m.team1 AS home_team,
    m.team2 AS away_team,
    m.status,
    CAST(NULL AS INTEGER) AS winner,
    m.winner AS actual_winner,
    m.team1_class,
    m.team2_class,
    m.match_class,
    COALESCE(o.match_win1, f.match_win1) AS reference_home_odds,
    COALESCE(o.match_win2, f.match_win2) AS reference_away_odds,
    COALESCE(o.set1_win1, f.set1_win1) AS reference_set_win1,
    COALESCE(o.set1_win2, f.set1_win2) AS reference_set_win2,
    lo.match_win1 AS home_odds,
    lo.match_win2 AS away_odds,
    lo.match_total_line AS match_total_line,
    lo.match_total_over AS match_total_over,
    lo.match_total_under AS match_total_under,
    lo.set_win1 AS set1_win1,
    lo.set_win2 AS set1_win2,
    lo.set_total_line AS set1_total_line,
    lo.set_total_over AS set1_total_over,
    lo.set_total_under AS set1_total_under,
    CAST(NULL AS REAL) AS home_match_serves,
    CAST(NULL AS REAL) AS home_match_serve_success,
    CAST(NULL AS REAL) AS home_match_serve_pct,
    CAST(NULL AS REAL) AS away_match_serves,
    CAST(NULL AS REAL) AS away_match_serve_success,
    CAST(NULL AS REAL) AS away_match_serve_pct,
    CAST(NULL AS REAL) AS completed_sets,
    CAST(NULL AS REAL) AS home_sets_won,
    CAST(NULL AS REAL) AS away_sets_won,
    CASE
        WHEN re.live_home_serve_volume > 0
        THEN CAST(re.live_home_serve_wins AS REAL) / re.live_home_serve_volume
        ELSE NULL
    END AS live_home_serve_pct,
    CASE
        WHEN re.live_away_serve_volume > 0
        THEN CAST(re.live_away_serve_wins AS REAL) / re.live_away_serve_volume
        ELSE NULL
    END AS live_away_serve_pct,
    CAST(re.live_home_serve_volume AS REAL) AS live_home_serve_volume,
    CAST(re.live_away_serve_volume AS REAL) AS live_away_serve_volume,
    'rally_live' AS odds_source,
    re.rally_db_id,
    re.set_number,
    re.rally_number,
    re.score1,
    re.score2,
    lo.ts AS odds_ts,
    lo.match_hcap_line,
    lo.match_hcap1,
    lo.match_hcap2,
    lo.set_hcap_line,
    lo.set_hcap1,
    lo.set_hcap2
FROM rallies_enriched re
INNER JOIN latest_rally_odds lo
    ON lo.rally_db_id = re.rally_db_id
INNER JOIN matches m
    ON m.id = re.match_id
LEFT JOIN opening_odds o
    ON o.match_id = m.id
LEFT JOIN first_seen_odds f
    ON f.match_id = m.id
WHERE m.id IN ({ids_sql})
  AND m.winner IN (1, 2)
  AND COALESCE(m.abandoned, 0) = 0
  AND lo.match_win1 IS NOT NULL
  AND lo.match_win2 IS NOT NULL
ORDER BY COALESCE(lo.ts, re.created_at) ASC, m.id ASC, re.rally_db_id ASC
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


def load_dataset_diagnostics(query: str = DATASET_DIAGNOSTICS_QUERY) -> pd.DataFrame:
    if not settings.db_url:
        raise ValueError("DB_URL is empty. Fill .env before loading diagnostics.")

    engine = create_engine(settings.db_url)
    with engine.connect() as connection:
        return pd.read_sql(text(query), connection)


def load_live_matches(query: str = LIVE_MATCHES_QUERY) -> pd.DataFrame:
    if not settings.db_url:
        raise ValueError("DB_URL is empty. Fill .env before loading live matches.")

    engine = create_engine(settings.db_url)
    with engine.connect() as connection:
        return pd.read_sql(text(query), connection)


def load_rally_backtest_snapshots(match_ids: list[int]) -> pd.DataFrame:
    if not settings.db_url:
        raise ValueError("DB_URL is empty. Fill .env before loading rally snapshots.")

    query = build_rally_backtest_query(match_ids)
    engine = create_engine(settings.db_url)
    with engine.connect() as connection:
        return pd.read_sql(text(query), connection)
