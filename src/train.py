from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from src.config import settings
from src.db import load_dataset_diagnostics, load_matches
from src.features import build_features


def train_and_evaluate(matches, label: str) -> dict[str, float | int | object]:
    x, y = build_features(matches)
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n=== {label} ===")
    print(f"Loaded rows for modeling: {len(matches)}")
    print(f"Training rows: {len(x_train)}")
    print(f"Test rows: {len(x_test)}")

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
