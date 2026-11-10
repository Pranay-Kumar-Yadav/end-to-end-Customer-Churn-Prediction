import pandas as pd
import numpy as np
import pytest

from src.preprocessing import build_preprocessor, split_features_target


def sample_data():
    return pd.DataFrame(
        {
            "customerID": ["A", "B", "C", "D"],
            "tenure": [1, 12, 24, 48],
            "TotalCharges": ["29.85", "346.8", "", "2000.0"],
            "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month"],
            "Churn": ["Yes", "No", "0", "1"],
        }
    )


def test_split_target_and_engineer_features():
    features, target = split_features_target(sample_data())
    assert "customerID" not in features
    assert features["TotalCharges"].dtype.kind in "fi"
    assert target.tolist() == [1, 0, 0, 1]


def test_preprocessor_handles_missing_numeric_and_unknown_categories():
    features, _ = split_features_target(sample_data())
    preprocessor = build_preprocessor(features)
    transformed = preprocessor.fit_transform(features.iloc[:3])
    assert transformed.shape[0] == 3
    assert transformed.shape[1] > 0
    values = transformed.toarray() if hasattr(transformed, "toarray") else transformed
    assert np.isfinite(values).all()


def test_rejects_unrecognized_target_values():
    data = sample_data()
    data.loc[0, "Churn"] = "maybe"
    with pytest.raises(ValueError, match="Unsupported churn labels"):
        split_features_target(data)
