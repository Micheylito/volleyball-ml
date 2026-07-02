from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.config import settings
from src.db import load_matches, load_rally_backtest_snapshots
from src.features import build_features, build_inference_features
from src.train import build_model, time_based_split


OUTPUT_DIR = Path("data/processed")
SIGNAL_THRESHOLD = 0.80
BATCH_MATCH_COUNT = 250
ODDS_BUCKETS = (1.20, 1.40, 1.70, 2.20)
PRIMARY_SUMMARY_NAME = "first_signal_per_match"
FOCUSED_ODDS_BUCKET = "1.40-1.69"
META_KEEP_THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70)
META_MIN_TRAIN_COVERAGE = 0.20
LIVE_FEATURE_COLUMNS = [
    "live_score_gap",
    "live_total_points",
    "live_set_number",
    "live_match_prob_shift_from_reference",
    "live_set_match_gap_delta",
]


def train_reference_model(train_matches: pd.DataFrame):
    x_train, y_train = build_features(train_matches, active_blocks=settings.feature_blocks)
    model = build_model(settings.model_family)
    model.fit(x_train, y_train)
    return model


def _chunk_match_ids(match_ids: list[int], chunk_size: int) -> list[list[int]]:
    return [
        match_ids[index : index + chunk_size]
        for index in range(0, len(match_ids), chunk_size)
    ]


def build_signal_rows(
    model,
    train_matches: pd.DataFrame,
    rally_snapshots: pd.DataFrame,
) -> pd.DataFrame:
    if rally_snapshots.empty:
        return rally_snapshots.copy()

    combined = pd.concat([train_matches, rally_snapshots], ignore_index=True)
    inference_features = build_inference_features(combined, active_blocks=settings.feature_blocks)
    snapshot_features = inference_features.tail(len(rally_snapshots)).reset_index(drop=True)

    probabilities = model.predict_proba(snapshot_features)[:, 1]
    output = rally_snapshots.reset_index(drop=True).copy()
    output["pred_home_win_proba"] = probabilities
    output["signal_side"] = pd.NA
    output.loc[output["pred_home_win_proba"] >= SIGNAL_THRESHOLD, "signal_side"] = "home"
    output.loc[output["pred_home_win_proba"] <= (1.0 - SIGNAL_THRESHOLD), "signal_side"] = "away"
    output = output[output["signal_side"].notna()].copy()

    if output.empty:
        return output

    output["signal_probability"] = output["pred_home_win_proba"]
    away_mask = output["signal_side"] == "away"
    output.loc[away_mask, "signal_probability"] = 1.0 - output.loc[away_mask, "pred_home_win_proba"]
    output["market_odds"] = output["home_odds"]
    output.loc[away_mask, "market_odds"] = output.loc[away_mask, "away_odds"]
    output["is_correct"] = 0
    output.loc[
        (output["signal_side"] == "home") & (output["actual_winner"] == 1),
        "is_correct",
    ] = 1
    output.loc[
        (output["signal_side"] == "away") & (output["actual_winner"] == 2),
        "is_correct",
    ] = 1
    output["snapshot_ts"] = pd.to_datetime(output["match_date"])
    return output.sort_values(
        ["snapshot_ts", "match_id", "set_number", "rally_number"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)


def build_signal_rows_in_batches(
    model,
    train_matches: pd.DataFrame,
    test_match_ids: list[int],
) -> tuple[pd.DataFrame, int]:
    signal_frames: list[pd.DataFrame] = []
    total_snapshots = 0
    match_id_batches = _chunk_match_ids(test_match_ids, BATCH_MATCH_COUNT)

    for batch_index, match_id_batch in enumerate(match_id_batches, start=1):
        rally_snapshots = load_rally_backtest_snapshots(match_id_batch)
        total_snapshots += len(rally_snapshots)
        print(
            f"Processing batch {batch_index}/{len(match_id_batches)}: "
            f"matches={len(match_id_batch)}, snapshots={len(rally_snapshots)}"
        )
        batch_signals = build_signal_rows(model, train_matches, rally_snapshots)
        print(f"  Signals in batch: {len(batch_signals)}")
        if not batch_signals.empty:
            signal_frames.append(batch_signals)

    if not signal_frames:
        return pd.DataFrame(), total_snapshots

    signal_rows = pd.concat(signal_frames, ignore_index=True)
    signal_rows = signal_rows.sort_values(
        ["snapshot_ts", "match_id", "set_number", "rally_number"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    return signal_rows, total_snapshots


def summarize_signals(signals: pd.DataFrame, summary_name: str) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame(
            columns=[
                "summary_name",
                "signal_side",
                "signal_rows",
                "unique_matches",
                "accuracy",
                "avg_probability",
                "avg_market_odds",
                "median_market_odds",
                "p10_market_odds",
                "p90_market_odds",
                "min_market_odds",
                "max_market_odds",
                "avg_set_number",
            ]
        )

    grouped = (
        signals.groupby("signal_side", dropna=False)
        .agg(
            signal_rows=("match_id", "count"),
            unique_matches=("match_id", "nunique"),
            accuracy=("is_correct", "mean"),
            avg_probability=("signal_probability", "mean"),
            avg_market_odds=("market_odds", "mean"),
            median_market_odds=("market_odds", "median"),
            p10_market_odds=("market_odds", lambda values: float(values.quantile(0.10))),
            p90_market_odds=("market_odds", lambda values: float(values.quantile(0.90))),
            min_market_odds=("market_odds", "min"),
            max_market_odds=("market_odds", "max"),
            avg_set_number=("set_number", "mean"),
        )
        .reset_index()
    )
    grouped.insert(0, "summary_name", summary_name)
    return grouped


def _label_odds_bucket(value: float) -> str:
    if value < ODDS_BUCKETS[0]:
        return "1.01-1.19"
    if value < ODDS_BUCKETS[1]:
        return "1.20-1.39"
    if value < ODDS_BUCKETS[2]:
        return "1.40-1.69"
    if value < ODDS_BUCKETS[3]:
        return "1.70-2.19"
    return "2.20+"


def summarize_signals_by_odds_bucket(signals: pd.DataFrame, summary_name: str) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame(
            columns=[
                "summary_name",
                "signal_side",
                "odds_bucket",
                "signal_rows",
                "unique_matches",
                "accuracy",
                "avg_probability",
                "avg_market_odds",
                "median_market_odds",
                "min_market_odds",
                "max_market_odds",
                "avg_set_number",
            ]
        )

    bucketed = signals.copy()
    bucketed["odds_bucket"] = bucketed["market_odds"].apply(_label_odds_bucket)
    grouped = (
        bucketed.groupby(["signal_side", "odds_bucket"], dropna=False)
        .agg(
            signal_rows=("match_id", "count"),
            unique_matches=("match_id", "nunique"),
            accuracy=("is_correct", "mean"),
            avg_probability=("signal_probability", "mean"),
            avg_market_odds=("market_odds", "mean"),
            median_market_odds=("market_odds", "median"),
            min_market_odds=("market_odds", "min"),
            max_market_odds=("market_odds", "max"),
            avg_set_number=("set_number", "mean"),
        )
        .reset_index()
    )
    grouped.insert(0, "summary_name", summary_name)
    return grouped.sort_values(
        ["signal_side", "odds_bucket"],
        ascending=[True, True],
    ).reset_index(drop=True)


def _normalized_home_probability(home_odds: pd.Series, away_odds: pd.Series) -> pd.Series:
    implied_home = 1.0 / pd.to_numeric(home_odds, errors="coerce")
    implied_away = 1.0 / pd.to_numeric(away_odds, errors="coerce")
    overround = implied_home + implied_away
    return implied_home / overround


def add_live_feature_columns(signals: pd.DataFrame) -> pd.DataFrame:
    if signals.empty:
        return signals.copy()

    enriched = signals.copy()
    current_home_prob = _normalized_home_probability(enriched["home_odds"], enriched["away_odds"])
    reference_home_prob = _normalized_home_probability(
        enriched["reference_home_odds"],
        enriched["reference_away_odds"],
    )
    current_set_home_prob = _normalized_home_probability(
        enriched["set1_win1"],
        enriched["set1_win2"],
    )

    enriched["odds_bucket"] = enriched["market_odds"].apply(_label_odds_bucket)
    enriched["live_score_gap"] = (
        pd.to_numeric(enriched["score1"], errors="coerce").fillna(0.0)
        - pd.to_numeric(enriched["score2"], errors="coerce").fillna(0.0)
    )
    enriched["live_total_points"] = (
        pd.to_numeric(enriched["score1"], errors="coerce").fillna(0.0)
        + pd.to_numeric(enriched["score2"], errors="coerce").fillna(0.0)
    )
    enriched["live_set_number"] = pd.to_numeric(
        enriched["set_number"], errors="coerce"
    ).fillna(0.0)
    enriched["live_match_prob_shift_from_reference"] = (
        current_home_prob - reference_home_prob
    ).fillna(0.0)
    enriched["live_set_match_gap_delta"] = (
        (current_set_home_prob - (1.0 - current_set_home_prob))
        - (current_home_prob - (1.0 - current_home_prob))
    ).fillna(0.0)
    return enriched


def build_first_signal_rows_in_batches(
    model,
    train_matches: pd.DataFrame,
    match_ids: list[int],
) -> tuple[pd.DataFrame, int]:
    signal_rows, total_snapshots = build_signal_rows_in_batches(model, train_matches, match_ids)
    if signal_rows.empty:
        return signal_rows, total_snapshots

    first_signal_rows = (
        signal_rows.sort_values(["snapshot_ts", "match_id", "set_number", "rally_number"])
        .drop_duplicates(subset=["match_id"], keep="first")
        .reset_index(drop=True)
    )
    return first_signal_rows, total_snapshots


def select_meta_threshold(train_zone: pd.DataFrame, probability_column: str) -> float:
    if train_zone.empty:
        return META_KEEP_THRESHOLDS[0]

    best_threshold = META_KEEP_THRESHOLDS[0]
    best_accuracy = -1.0
    best_coverage = -1.0
    total_rows = len(train_zone)

    for threshold in META_KEEP_THRESHOLDS:
        kept = train_zone[train_zone[probability_column] >= threshold]
        if kept.empty:
            continue
        coverage = len(kept) / total_rows
        if coverage < META_MIN_TRAIN_COVERAGE:
            continue
        accuracy = float(kept["is_correct"].mean())
        if (accuracy, coverage) > (best_accuracy, best_coverage):
            best_threshold = threshold
            best_accuracy = accuracy
            best_coverage = coverage

    return best_threshold


def summarize_meta_strategy(
    rows: pd.DataFrame,
    strategy_name: str,
    probability_column: str,
    keep_threshold: float,
) -> dict[str, float | int | str]:
    kept = rows[rows[probability_column] >= keep_threshold].copy()
    if kept.empty:
        return {
            "strategy_name": strategy_name,
            "keep_threshold": keep_threshold,
            "matches": 0,
            "coverage": 0.0,
            "accuracy": 0.0,
            "avg_market_odds": 0.0,
        }

    return {
        "strategy_name": strategy_name,
        "keep_threshold": keep_threshold,
        "matches": len(kept),
        "coverage": len(kept) / len(rows),
        "accuracy": float(kept["is_correct"].mean()),
        "avg_market_odds": float(kept["market_odds"].mean()),
    }


def run_meta_filter_analysis(
    train_first_signals: pd.DataFrame,
    test_first_signals: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_signals = add_live_feature_columns(train_first_signals)
    test_signals = add_live_feature_columns(test_first_signals)

    train_zone = train_signals[train_signals["odds_bucket"] == FOCUSED_ODDS_BUCKET].copy()
    test_zone = test_signals[test_signals["odds_bucket"] == FOCUSED_ODDS_BUCKET].copy()

    results: list[dict[str, float | int | str]] = []
    if test_zone.empty:
        return pd.DataFrame(), pd.DataFrame()
    if train_zone.empty or train_zone["is_correct"].nunique() < 2:
        return pd.DataFrame(), pd.DataFrame()

    results.append(
        {
            "strategy_name": "bet_all",
            "keep_threshold": 0.0,
            "matches": len(test_zone),
            "coverage": 1.0,
            "accuracy": float(test_zone["is_correct"].mean()),
            "avg_market_odds": float(test_zone["market_odds"].mean()),
        }
    )

    probability_only_model = LogisticRegression(max_iter=1000)
    probability_only_model.fit(
        train_zone[["signal_probability"]].fillna(0.0),
        train_zone["is_correct"],
    )
    train_signals["meta_probability_only"] = probability_only_model.predict_proba(
        train_signals[["signal_probability"]].fillna(0.0)
    )[:, 1]
    test_signals["meta_probability_only"] = probability_only_model.predict_proba(
        test_signals[["signal_probability"]].fillna(0.0)
    )[:, 1]

    enhanced_columns = ["signal_probability", *LIVE_FEATURE_COLUMNS]
    enhanced_model = LogisticRegression(max_iter=1000)
    enhanced_model.fit(
        train_zone[enhanced_columns].fillna(0.0),
        train_zone["is_correct"],
    )
    train_signals["meta_live_enhanced"] = enhanced_model.predict_proba(
        train_signals[enhanced_columns].fillna(0.0)
    )[:, 1]
    test_signals["meta_live_enhanced"] = enhanced_model.predict_proba(
        test_signals[enhanced_columns].fillna(0.0)
    )[:, 1]

    train_zone = train_signals[train_signals["odds_bucket"] == FOCUSED_ODDS_BUCKET].copy()
    test_zone = test_signals[test_signals["odds_bucket"] == FOCUSED_ODDS_BUCKET].copy()

    probability_only_threshold = select_meta_threshold(train_zone, "meta_probability_only")
    enhanced_threshold = select_meta_threshold(train_zone, "meta_live_enhanced")

    results.append(
        summarize_meta_strategy(
            test_zone,
            "probability_only_filter",
            "meta_probability_only",
            probability_only_threshold,
        )
    )
    results.append(
        summarize_meta_strategy(
            test_zone,
            "live_enhanced_filter",
            "meta_live_enhanced",
            enhanced_threshold,
        )
    )

    result_frame = pd.DataFrame(results)
    feature_importance_frame = pd.DataFrame(
        {
            "feature": ["signal_probability", *LIVE_FEATURE_COLUMNS],
            "coefficient": enhanced_model.coef_[0],
        }
    ).sort_values("coefficient", ascending=False, key=lambda values: values.abs())
    return result_frame, feature_importance_frame


def main() -> None:
    print(f"Rally backtest model family: {settings.model_family}")
    print(f"Rally backtest feature blocks: {', '.join(settings.feature_blocks)}")
    print(f"Signal threshold: {SIGNAL_THRESHOLD:.0%}")

    matches = load_matches()
    train_matches, test_matches = time_based_split(matches)
    model = train_reference_model(train_matches)

    test_match_ids = test_matches["match_id"].astype(int).tolist()
    print(f"Train matches: {len(train_matches)}")
    print(f"Test matches: {len(test_matches)}")
    print(f"Batch match count: {BATCH_MATCH_COUNT}")

    signal_rows, total_snapshots = build_signal_rows_in_batches(
        model,
        train_matches,
        test_match_ids,
    )
    print(f"Loaded rally snapshots: {total_snapshots}")
    print(f"Signal rows above threshold: {len(signal_rows)}")

    first_signal_rows = (
        signal_rows.sort_values(["snapshot_ts", "match_id", "set_number", "rally_number"])
        .drop_duplicates(subset=["match_id"], keep="first")
        .reset_index(drop=True)
    )
    train_match_ids = train_matches["match_id"].astype(int).tolist()
    train_first_signal_rows, train_total_snapshots = build_first_signal_rows_in_batches(
        model,
        train_matches,
        train_match_ids,
    )
    print(f"Loaded train rally snapshots: {train_total_snapshots}")
    print(f"Train first-signal rows: {len(train_first_signal_rows)}")

    summary_all = summarize_signals(signal_rows, "all_signal_rows")
    summary_first = summarize_signals(first_signal_rows, "first_signal_per_match")
    combined_summary = pd.concat([summary_all, summary_first], ignore_index=True)
    bucket_summary = summarize_signals_by_odds_bucket(
        first_signal_rows,
        PRIMARY_SUMMARY_NAME,
    )
    meta_summary, meta_feature_importance = run_meta_filter_analysis(
        train_first_signal_rows,
        first_signal_rows,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    signal_rows_path = OUTPUT_DIR / "rally_backtest_signal_rows.csv"
    first_signal_rows_path = OUTPUT_DIR / "rally_backtest_first_signal_rows.csv"
    summary_path = OUTPUT_DIR / "rally_backtest_signal_summary.csv"
    bucket_summary_path = OUTPUT_DIR / "rally_backtest_first_signal_odds_buckets.csv"
    meta_summary_path = OUTPUT_DIR / "rally_backtest_meta_zone_140_169_summary.csv"
    meta_features_path = OUTPUT_DIR / "rally_backtest_meta_zone_140_169_features.csv"

    signal_rows.to_csv(signal_rows_path, index=False)
    first_signal_rows.to_csv(first_signal_rows_path, index=False)
    combined_summary.to_csv(summary_path, index=False)
    bucket_summary.to_csv(bucket_summary_path, index=False)
    meta_summary.to_csv(meta_summary_path, index=False)
    meta_feature_importance.to_csv(meta_features_path, index=False)

    print("Rally signal summary:")
    if combined_summary.empty:
        print("  No signals found above threshold.")
    else:
        for row in combined_summary.itertuples(index=False):
            print(
                f"  {row.summary_name} | {row.signal_side}: "
                f"rows={row.signal_rows}, matches={row.unique_matches}, "
                f"accuracy={row.accuracy:.4f}, avg_odds={row.avg_market_odds:.2f}, "
                f"median_odds={row.median_market_odds:.2f}, "
                f"p10={row.p10_market_odds:.2f}, p90={row.p90_market_odds:.2f}, "
                f"range={row.min_market_odds:.2f}-{row.max_market_odds:.2f}"
            )

    print(f"\nPrimary mode: {PRIMARY_SUMMARY_NAME} (1 bet = 1 match)")
    print("Odds bucket summary:")
    if bucket_summary.empty:
        print("  No first-signal rows found above threshold.")
    else:
        for row in bucket_summary.itertuples(index=False):
            print(
                f"  {row.signal_side} | {row.odds_bucket}: "
                f"rows={row.signal_rows}, matches={row.unique_matches}, "
                f"accuracy={row.accuracy:.4f}, avg_odds={row.avg_market_odds:.2f}, "
                f"median_odds={row.median_market_odds:.2f}, "
                f"range={row.min_market_odds:.2f}-{row.max_market_odds:.2f}"
            )

    print(f"\nFocused zone check: {FOCUSED_ODDS_BUCKET}")
    if meta_summary.empty:
        print("  No rows available for focused meta analysis.")
    else:
        for row in meta_summary.itertuples(index=False):
            print(
                f"  {row.strategy_name}: matches={row.matches}, "
                f"coverage={row.coverage:.4f}, accuracy={row.accuracy:.4f}, "
                f"avg_odds={row.avg_market_odds:.2f}, keep_threshold={row.keep_threshold:.2f}"
            )
    if not meta_feature_importance.empty:
        print("  Enhanced live feature coefficients:")
        for row in meta_feature_importance.itertuples(index=False):
            print(f"    {row.feature}: {row.coefficient:.4f}")

    print(f"Signal rows saved to {signal_rows_path}")
    print(f"First signal rows saved to {first_signal_rows_path}")
    print(f"Signal summary saved to {summary_path}")
    print(f"First-signal odds buckets saved to {bucket_summary_path}")
    print(f"Focused meta summary saved to {meta_summary_path}")
    print(f"Focused meta features saved to {meta_features_path}")


if __name__ == "__main__":
    main()
