from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.load_model import load_toxicity_model
from src.prediction import (
    predict_toxicity,
    predict_batch_binary,
    predict_batch,
    get_model_and_tokenizer,
)
from src.utils import (
    LABELS,
    POLICY_PRESETS,
    OPTIMAL_THRESHOLDS,
)

app = FastAPI(
    title="Comment Toxicity Detection API",
    description="Multi-label toxicity classification API powered by fine-tuned DistilBERT",
    version="2.0.0",
)


class SingleCommentRequest(BaseModel):
    comment: str = Field(..., example="You are a disgusting idiot")
    policy: Optional[str] = Field("Balanced (F1-Optimized)", example="Balanced (F1-Optimized)")
    custom_thresholds: Optional[Dict[str, float]] = None


class BatchCommentsRequest(BaseModel):
    comments: List[str] = Field(..., example=["Have a nice day", "I will hurt you"])
    policy: Optional[str] = Field("Balanced (F1-Optimized)", example="Balanced (F1-Optimized)")
    custom_thresholds: Optional[Dict[str, float]] = None


@app.on_event("startup")
def startup_event():
    # Warm up model weights on startup
    get_model_and_tokenizer()


@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Comment Toxicity Detection API",
        "model": "DistilBERT Multi-Label",
        "endpoints": ["/predict", "/predict/batch", "/policies"],
    }


@app.get("/policies")
def get_policies():
    """Return available moderation policies and default thresholds."""
    return {
        "available_policies": list(POLICY_PRESETS.keys()),
        "optimal_thresholds": OPTIMAL_THRESHOLDS,
        "labels": LABELS,
    }


@app.post("/predict")
def predict_single(req: SingleCommentRequest):
    """Predict toxicity categories and probabilities for a single comment."""
    if not req.comment.strip():
        raise HTTPException(status_code=400, detail="Comment text cannot be empty.")

    thresholds = req.custom_thresholds or POLICY_PRESETS.get(
        req.policy, OPTIMAL_THRESHOLDS
    )

    results = predict_toxicity(req.comment, threshold=thresholds)
    return {
        "comment": req.comment,
        "policy": req.policy,
        "results": results,
    }


@app.post("/predict/batch")
def predict_batch_endpoint(req: BatchCommentsRequest):
    """Batch predict toxicity for multiple comments."""
    if not req.comments:
        raise HTTPException(status_code=400, detail="Comments list cannot be empty.")

    thresholds = req.custom_thresholds or POLICY_PRESETS.get(
        req.policy, OPTIMAL_THRESHOLDS
    )

    batch_results = predict_batch(req.comments, threshold=thresholds)
    binary_predictions = predict_batch_binary(req.comments, threshold=thresholds)

    return {
        "total_analyzed": len(req.comments),
        "policy": req.policy,
        "labels": LABELS,
        "detailed_results": batch_results,
        "binary_matrix": binary_predictions,
    }
