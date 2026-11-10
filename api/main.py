from functools import lru_cache

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import MODEL_PATH, PREPROCESSOR_PATH
from src.preprocessing import prepare_prediction_features

app = FastAPI(title="Customer Churn Prediction API", version="1.0.0")


class PredictionRequest(BaseModel):
    features: dict[str, object] = Field(..., min_length=1)


@lru_cache(maxsize=1)
def load_artifacts():
    if not MODEL_PATH.is_file() or not PREPROCESSOR_PATH.is_file():
        raise FileNotFoundError("Model artifacts are missing. Run `python -m src.train` first.")
    return joblib.load(MODEL_PATH), joblib.load(PREPROCESSOR_PATH)


@app.get("/health")
def health():
    ready = MODEL_PATH.is_file() and PREPROCESSOR_PATH.is_file()
    return {"status": "ready" if ready else "model_not_trained"}


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        model, preprocessor = load_artifacts()
        features = prepare_prediction_features(pd.DataFrame([request.features]))
        transformed = preprocessor.transform(features)
        probability = float(model.predict_proba(transformed)[0, 1])
        return {"churn": bool(probability >= 0.5), "churn_probability": probability}
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except (ValueError, KeyError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
