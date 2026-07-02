# Current Baseline

Baseline date: 2026-07-02

Current validation approach:

- time-based split
- train on older matches
- test on newer matches
- compare `full_coverage` vs `odds_only`

Default feature blocks:

- `core_market`
- `market_derived`
- `rest`
- `form_base`
- `serve_form`
- `league_form`
- `context_form`
- `live_serve`

Latest observed dataset snapshot:

- all matches: `24781`
- trainable matches: `22175`
- trainable with any odds: `18932`
- trainable with match serve summary: `20184`
- trainable with completed sets: `22165`

Latest model comparison:

`full_coverage`

- rows: `22175`
- accuracy: `0.6868`
- f1_macro: `0.6761`
- train date range: `2025-12-17 04:25:20 -> 2026-04-29 13:31:30`
- test date range: `2026-04-29 13:32:28 -> 2026-07-01 15:20:51`

`odds_only`

- rows: `18932`
- accuracy: `0.6807`
- f1_macro: `0.6735`
- train date range: `2025-12-17 04:25:20 -> 2026-04-30 08:29:44`
- test date range: `2026-04-30 08:51:08 -> 2026-07-01 15:20:51`

Current selected model:

- `full_coverage`
- selection rule: max `f1_macro`, then `accuracy`

Interpretation:

- `market_derived` is the first experimental block that produced a real uplift
- `full_coverage` is again the best mode on time-based validation
- `set_trends` remains available as an experiment, but is not in the default baseline
- new features should be judged against this baseline
