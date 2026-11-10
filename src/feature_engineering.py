import pandas as pd


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Apply stable, schema-light feature cleanup used for training and inference."""
    features = data.copy()
    for column in features.columns:
        if column.strip().lower() == "totalcharges":
            features[column] = pd.to_numeric(features[column], errors="coerce")
    return features
