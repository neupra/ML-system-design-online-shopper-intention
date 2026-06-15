from typing import List

from fastapi import FastAPI, HTTPException

from serving.app.schemas import ShopperInput
from serving.app.model_loader import (
    load_model,
    is_model_loaded,
    predict_records,
    MODEL_PATH,
)


app = FastAPI(
    title="Online Shoppers Revenue Prediction API",
    description="FastAPI serving pipeline using joblib inference model",
    version="1.0.0",
)


@app.on_event("startup")
def startup_event():
    load_model()


def payload_to_dict(payload):
    if hasattr(payload, "model_dump"):
        return payload.model_dump()
    return payload.dict()


@app.get("/")
def root():
    return {
        "message": "Online Shoppers Revenue Prediction API is running."
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": is_model_loaded(),
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict")
def predict(payload: ShopperInput):
    try:
        record = payload_to_dict(payload)
        prediction = predict_records([record])[0]
        return prediction

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict_batch")
def predict_batch(payload: List[ShopperInput]):
    try:
        records = [payload_to_dict(item) for item in payload]
        predictions = predict_records(records)

        return {
            "count": len(predictions),
            "predictions": predictions,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))