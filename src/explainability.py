from pathlib import Path

import joblib
import pandas as pd

from src.config import MODEL_PATH, PREPROCESSOR_PATH


def feature_importance(
    model_path: str | Path = MODEL_PATH,
    preprocessor_path: str | Path = PREPROCESSOR_PATH,
) -> pd.DataFrame:
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    names = preprocessor.get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = model.coef_[0]
    else:
        raise ValueError("The saved estimator does not expose feature importance or coefficients.")
    return pd.DataFrame({"feature": names, "importance": values}).assign(
        absolute_importance=lambda frame: frame["importance"].abs()
    ).sort_values("absolute_importance", ascending=False, ignore_index=True)
