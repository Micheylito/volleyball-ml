from __future__ import annotations

from typing import Any

import joblib
import pandas as pd

from src.config import settings


def predict_match(features: dict[str, Any]) -> float:
    model = joblib.load(settings.model_path)
    sample = pd.DataFrame([features])

    # Keep inference compatible with whatever feature schema the saved model
    # was trained on. Extra columns are dropped, missing ones are filled with 0.
    trained_columns = getattr(model, "feature_names_in_", None)
    if trained_columns is not None:
        for column in trained_columns:
            if column not in sample.columns:
                sample[column] = 0.0
        sample = sample[list(trained_columns)]

    probability = model.predict_proba(sample)[0][1]
    return float(probability)
