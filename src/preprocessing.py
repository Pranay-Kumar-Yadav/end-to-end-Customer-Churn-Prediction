import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import ID_COLUMNS, TARGET_COLUMN
from src.feature_engineering import engineer_features


def _find_target(data: pd.DataFrame, target_column: str = TARGET_COLUMN) -> str:
    matches = [column for column in data.columns if column.lower() == target_column.lower()]
    if not matches:
        raise ValueError(
            f"Target column '{target_column}' was not found. "
            f"Available columns: {', '.join(map(str, data.columns))}"
        )
    return matches[0]


def split_features_target(
    data: pd.DataFrame, target_column: str = TARGET_COLUMN
) -> tuple[pd.DataFrame, pd.Series]:
    target_name = _find_target(data, target_column)
    target = data[target_name]
    labels = target.astype(str).str.strip().str.lower().map(
        {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0}
    )
    if labels.isna().any():
        invalid = sorted(target[labels.isna()].astype(str).unique())
        raise ValueError(f"Unsupported churn labels: {invalid}. Expected yes/no or 1/0.")

    features = data.drop(columns=[target_name])
    features = features.drop(columns=[column for column in ID_COLUMNS if column in features])
    features = engineer_features(features)
    return features, labels.astype("int8").rename(target_name)


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = [column for column in features.columns if column not in numeric_columns]
    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            )
        )
    if not transformers:
        raise ValueError("No feature columns remain after removing the target and identifiers.")
    return ColumnTransformer(transformers=transformers, remainder="drop")


def prepare_prediction_features(data: pd.DataFrame) -> pd.DataFrame:
    features = data.drop(columns=[column for column in ID_COLUMNS if column in data])
    return engineer_features(features)
