from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_PATH


def load_data(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load a non-empty customer churn CSV."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Add customer_churn.csv to data/raw/."
        )
    data = pd.read_csv(csv_path)
    if data.empty:
        raise ValueError(f"Dataset at {csv_path} contains no rows.")
    if data.columns.has_duplicates:
        raise ValueError("Dataset contains duplicate column names.")
    return data
