# Current Baseline

Baseline date: 2026-07-01

Current validation approach:

- time-based split
- train on older matches
- test on newer matches
- compare `full_coverage` vs `odds_only`

Current feature groups:

- opening odds
- first seen odds fallback
- team classes
- match metadata
- rolling form windows `3/5/10`
- rest days

Latest observed dataset snapshot:

- all matches: `24779`
- trainable matches: `22175`
- trainable with any odds: `18932`

Latest model comparison:

`full_coverage`

- rows: `22175`
- accuracy: `0.6843`
- f1_macro: `0.6730`
- train date range: `2025-12-17 04:25:20 -> 2026-04-29 13:31:30`
- test date range: `2026-04-29 13:32:28 -> 2026-07-01 15:20:51`

`odds_only`

- rows: `18932`
- accuracy: `0.6818`
- f1_macro: `0.6739`
- train date range: `2025-12-17 04:25:20 -> 2026-04-30 08:29:44`
- test date range: `2026-04-30 08:51:08 -> 2026-07-01 15:20:51`

Current selected model:

- `odds_only`
- selection rule: max `f1_macro`, then `accuracy`

Interpretation:

- the project has a working ML baseline
- random split was too optimistic
- time-based validation is the reference going forward
- new features should be judged against this file
