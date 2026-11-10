# Customer Churn Prediction

A Python project for exploring customer churn data, preparing features, comparing classifiers, explaining model behavior, and serving predictions over HTTP.

## Dataset

Put a labeled CSV at `data/raw/customer_churn.csv`. The default target column is `Churn`; values can be `Yes`/`No`, `True`/`False`, or `1`/`0`. The starter CSV contains only the expected Telco dataset header because no source data was supplied. Replace it with real observations before training. Customer ID columns are excluded from model inputs, and `TotalCharges` is parsed as numeric.

## Setup and run

Use Python 3.10 or newer, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the automated checks and train/evaluate the candidate models:

```powershell
python -m pytest
python -m src.train
```

Training writes the processed feature table, model comparison metrics, and selected model/preprocessor artifacts under `data/processed/`, `reports/`, and `models/`. The model is selected by held-out ROC AUC.

## API

Start the service after training:

```powershell
uvicorn api.main:app --reload
```

Health check: `GET /health`  
Prediction: `POST /predict`

```json
{
  "features": {
    "gender": "Female",
    "SeniorCitizen": 0,
    "tenure": 12,
    "Contract": "Month-to-month",
    "MonthlyCharges": 70.35,
    "TotalCharges": "844.2"
  }
}
```

Supply the same feature names used by the training data, excluding `Churn` and any customer identifier. The response includes the predicted churn flag and probability. `GET /docs` provides interactive API documentation.

## Docker

Train locally first so the generated model files exist, then from `deployment/` run:

```powershell
docker compose up --build
```

## Notebooks

Run notebooks in order from `notebooks/` using a kernel with `requirements.txt` installed. The EDA notebook expects the raw CSV; the training and explainability notebooks expect the dataset and generated model artifacts.
