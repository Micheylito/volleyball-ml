from __future__ import annotations

from typing import Any

import joblib
import pandas as pd

from src.config import settings


def predict_match(features: dict[str, Any]) -> float:
    model = joblib.load(settings.model_path)
    sample = pd.DataFrame([features])
    probability = model.predict_proba(sample)[0][1]
    return float(probability)

