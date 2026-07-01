from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from src.config import settings
from src.db import load_dataset_diagnostics, load_matches
from src.features import build_features


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

    x, y = build_features(matches)
    print(f"Loaded rows for modeling: {len(matches)}")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training rows: {len(x_train)}")
    print(f"Test rows: {len(x_test)}")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    print(classification_report(y_test, predictions))

    model_path = Path(settings.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
