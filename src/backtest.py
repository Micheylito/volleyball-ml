from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.config import settings
from src.db import load_matches
from src.features import build_features
from src.train import build_model, time_based_split


OUTPUT_DIR = Path("data/processed")
LEAGUE_SELECTION_MODE = "odds_only"
MIN_LEAGUE_MATCHES = 30
MIN_LEAGUE_ACCURACY = 0.75
SIGNAL_THRESHOLDS = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80)


def build_test_predictions(matches: pd.DataFrame, label: str) -> tuple[pd.DataFrame, dict[str, float | int]]:
    train_matches, test_matches = time_based_split(matches)
    combined_matches = pd.concat([train_matches, test_matches], ignore_index=True)

    x_all, y_all = build_features(combined_matches, active_blocks=settings.feature_blocks)
    train_rows = len(train_matches)
    x_train = x_all.iloc[:train_rows]
    y_train = y_all.iloc[:train_rows]
    x_test = x_all.iloc[train_rows:]
    y_test = y_all.iloc[train_rows:]

    model = build_model(settings.model_family)
    model.fit(x_train, y_train)

    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    test_output = test_matches.reset_index(drop=True).copy()
    test_output["target_home_win"] = y_test.reset_index(drop=True)
    test_output["pred_home_win"] = predictions
    test_output["pred_home_win_proba"] = probabilities
    test_output["is_correct"] = (
        test_output["target_home_win"] == test_output["pred_home_win"]
    ).astype(int)
    test_output["mode"] = label

    summary = {
        "rows": len(matches),
        "train_rows": len(train_matches),
        "test_rows": len(test_matches),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1_macro": float(f1_score(y_test, predictions, average="macro")),
    }
    return test_output, summary


def build_group_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        predictions.groupby(["mode", "league"], dropna=False)
        .agg(
            matches=("match_id", "count"),
            accuracy=("is_correct", "mean"),
            avg_home_win_proba=("pred_home_win_proba", "mean"),
            avg_home_odds=("home_odds", "mean"),
            avg_away_odds=("away_odds", "mean"),
        )
        .reset_index()
        .sort_values(["mode", "matches", "accuracy"], ascending=[True, False, False])
    )
    return grouped


def build_allowed_leagues(league_summary: pd.DataFrame) -> pd.DataFrame:
    allowed = league_summary[
        (league_summary["mode"] == LEAGUE_SELECTION_MODE)
        & (league_summary["matches"] >= MIN_LEAGUE_MATCHES)
        & (league_summary["accuracy"] >= MIN_LEAGUE_ACCURACY)
    ].copy()
    allowed = allowed.sort_values(
        ["accuracy", "matches", "league"], ascending=[False, False, True]
    ).reset_index(drop=True)
    allowed["selection_mode"] = LEAGUE_SELECTION_MODE
    allowed["min_matches"] = MIN_LEAGUE_MATCHES
    allowed["min_accuracy"] = MIN_LEAGUE_ACCURACY
    return allowed


def build_signal_threshold_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for mode in sorted(predictions["mode"].dropna().unique()):
        mode_predictions = predictions[predictions["mode"] == mode].copy()
        for threshold in SIGNAL_THRESHOLDS:
            home_signals = mode_predictions[mode_predictions["pred_home_win_proba"] >= threshold].copy()
            away_signals = mode_predictions[mode_predictions["pred_home_win_proba"] <= (1.0 - threshold)].copy()

            if not home_signals.empty:
                rows.append(
                    {
                        "mode": mode,
                        "threshold": threshold,
                        "signal_side": "home",
                        "signals": len(home_signals),
                        "coverage": len(home_signals) / len(mode_predictions),
                        "accuracy": float((home_signals["target_home_win"] == 1).mean()),
                        "avg_probability": float(home_signals["pred_home_win_proba"].mean()),
                        "avg_market_odds": float(home_signals["home_odds"].mean()),
                        "min_market_odds": float(home_signals["home_odds"].min()),
                        "max_market_odds": float(home_signals["home_odds"].max()),
                    }
                )
            else:
                rows.append(
                    {
                        "mode": mode,
                        "threshold": threshold,
                        "signal_side": "home",
                        "signals": 0,
                        "coverage": 0.0,
                        "accuracy": 0.0,
                        "avg_probability": 0.0,
                        "avg_market_odds": 0.0,
                        "min_market_odds": 0.0,
                        "max_market_odds": 0.0,
                    }
                )

            if not away_signals.empty:
                rows.append(
                    {
                        "mode": mode,
                        "threshold": threshold,
                        "signal_side": "away",
                        "signals": len(away_signals),
                        "coverage": len(away_signals) / len(mode_predictions),
                        "accuracy": float((away_signals["target_home_win"] == 0).mean()),
                        "avg_probability": float((1.0 - away_signals["pred_home_win_proba"]).mean()),
                        "avg_market_odds": float(away_signals["away_odds"].mean()),
                        "min_market_odds": float(away_signals["away_odds"].min()),
                        "max_market_odds": float(away_signals["away_odds"].max()),
                    }
                )
            else:
                rows.append(
                    {
                        "mode": mode,
                        "threshold": threshold,
                        "signal_side": "away",
                        "signals": 0,
                        "coverage": 0.0,
                        "accuracy": 0.0,
                        "avg_probability": 0.0,
                        "avg_market_odds": 0.0,
                        "min_market_odds": 0.0,
                        "max_market_odds": 0.0,
                    }
                )

    return pd.DataFrame(rows)


def main() -> None:
    print(f"Backtest model family: {settings.model_family}")
    print(f"Backtest feature blocks: {', '.join(settings.feature_blocks)}")
    matches = load_matches()

    full_predictions, full_summary = build_test_predictions(matches, "full_coverage")
    odds_only_matches = matches[matches["odds_source"] != "missing"].copy()
    odds_predictions, odds_summary = build_test_predictions(odds_only_matches, "odds_only")

    all_predictions = pd.concat([full_predictions, odds_predictions], ignore_index=True)
    league_summary = build_group_summary(all_predictions)
    mode_summary = pd.DataFrame(
        [
            {"mode": "full_coverage", **full_summary},
            {"mode": "odds_only", **odds_summary},
        ]
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions_path = OUTPUT_DIR / "backtest_predictions.csv"
    league_summary_path = OUTPUT_DIR / "backtest_league_summary.csv"
    mode_summary_path = OUTPUT_DIR / "backtest_mode_summary.csv"
    signal_summary_path = OUTPUT_DIR / "backtest_signal_threshold_summary.csv"
    allowed_leagues_path = OUTPUT_DIR / "allowed_leagues.csv"
    allowed_leagues = build_allowed_leagues(league_summary)
    signal_summary = build_signal_threshold_summary(all_predictions)

    all_predictions.to_csv(predictions_path, index=False)
    league_summary.to_csv(league_summary_path, index=False)
    mode_summary.to_csv(mode_summary_path, index=False)
    signal_summary.to_csv(signal_summary_path, index=False)
    allowed_leagues.to_csv(allowed_leagues_path, index=False)

    print("Backtest mode summary:")
    for row in mode_summary.itertuples(index=False):
        print(
            f"  {row.mode}: rows={row.rows}, test_rows={row.test_rows}, "
            f"accuracy={row.accuracy:.4f}, f1_macro={row.f1_macro:.4f}"
        )

    print(f"Predictions saved to {predictions_path}")
    print(f"League summary saved to {league_summary_path}")
    print(f"Mode summary saved to {mode_summary_path}")
    print(f"Signal threshold summary saved to {signal_summary_path}")
    print("Signal threshold highlights:")
    for row in signal_summary[signal_summary["signals"] > 0].sort_values(
        ["accuracy", "signals"], ascending=[False, False]
    ).head(12).itertuples(index=False):
        print(
            f"  {row.mode} {row.signal_side} >= {row.threshold:.2f}: "
            f"signals={row.signals}, accuracy={row.accuracy:.4f}, coverage={row.coverage:.4f}, "
            f"avg_odds={row.avg_market_odds:.2f}, "
            f"range={row.min_market_odds:.2f}-{row.max_market_odds:.2f}"
        )
    print(
        f"Allowed leagues ({LEAGUE_SELECTION_MODE}, min_matches={MIN_LEAGUE_MATCHES}, "
        f"min_accuracy={MIN_LEAGUE_ACCURACY:.2f}): {len(allowed_leagues)}"
    )
    if not allowed_leagues.empty:
        for row in allowed_leagues.head(20).itertuples(index=False):
            print(f"  {row.league}: matches={row.matches}, accuracy={row.accuracy:.4f}")
    print(f"Allowed leagues saved to {allowed_leagues_path}")


if __name__ == "__main__":
    main()
