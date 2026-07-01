from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report

from src.config import settings
from src.db import load_dataset_diagnostics, load_matches
from src.features import build_features


def time_based_split(matches, test_size: float = 0.2):
    sorted_matches = matches.sort_values("match_date").reset_index(drop=True)
    split_index = int(len(sorted_matches) * (1 - test_size))

    if split_index <= 0 or split_index >= len(sorted_matches):
        raise ValueError("Not enough rows for a time-based split.")

    train_matches = sorted_matches.iloc[:split_index].copy()
    test_matches = sorted_matches.iloc[split_index:].copy()
    return train_matches, test_matches


def train_and_evaluate(matches, label: str) -> dict[str, float | int | object]:
    train_matches, test_matches = time_based_split(matches)
    combined_matches = pd.concat([train_matches, test_matches], ignore_index=True)

    x_all, y_all = build_features(combined_matches)
    train_rows = len(train_matches)
    x_train = x_all.iloc[:train_rows]
    y_train = y_all.iloc[:train_rows]
    x_test = x_all.iloc[train_rows:]
    y_test = y_all.iloc[train_rows:]

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
    )

    print(f"\n=== {label} ===")
    print(f"Loaded rows for modeling: {len(matches)}")
    print(f"Training rows: {len(x_train)}")
    print(f"Test rows: {len(x_test)}")
    print(
        f"Train date range: {train_matches['match_date'].min()} -> "
        f"{train_matches['match_date'].max()}"
    )
    print(
        f"Test date range: {test_matches['match_date'].min()} -> "
        f"{test_matches['match_date'].max()}"
    )

    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    print(classification_report(y_test, predictions))

    return {
        "label": label,
        "rows": len(matches),
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1_macro": float(f1_score(y_test, predictions, average="macro")),
        "model": model,
    }


def main() -> None:
    diagnostics = load_dataset_diagnostics()
    print("Dataset diagnostics:")
    for row in diagnostics.itertuples(index=False):
        print(f"  {row.metric}: {row.value}")

    matches = load_matches()
    odds_source_counts = matches["odds_source"].value_counts(dropna=False).to_dict()
    print("Odds source distribution:")
    for source, count in odds_source_counts.items():
        print(f"  {source}: {count}")

    full_result = train_and_evaluate(matches, "full_coverage")

    odds_only_matches = matches[matches["odds_source"] != "missing"].copy()
    odds_only_result = train_and_evaluate(odds_only_matches, "odds_only")

    model_path = Path(settings.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    best_result = max(
        [full_result, odds_only_result],
        key=lambda result: (result["f1_macro"], result["accuracy"]),
    )
    joblib.dump(best_result["model"], model_path)

    print("\nComparison summary:")
    for result in [full_result, odds_only_result]:
        print(
            f"  {result['label']}: rows={result['rows']}, "
            f"accuracy={result['accuracy']:.4f}, f1_macro={result['f1_macro']:.4f}"
        )

    print(
        f"Selected model: {best_result['label']} "
        f"(accuracy={best_result['accuracy']:.4f}, f1_macro={best_result['f1_macro']:.4f})"
    )
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
