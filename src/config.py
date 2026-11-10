from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "customer_churn.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "churn_processed.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"
PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "preprocessor.joblib"
COMPARISON_PATH = PROJECT_ROOT / "reports" / "model_comparison.csv"
TARGET_COLUMN = "Churn"
ID_COLUMNS = ("customerID", "customer_id", "CustomerID")
RANDOM_STATE = 42
