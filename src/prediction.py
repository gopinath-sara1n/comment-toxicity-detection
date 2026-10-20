import torch
from src.load_model import load_toxicity_model
from src.preprocessing import clean_text
from src.utils import (
    LABELS,
    OPTIMAL_THRESHOLDS,
    calculate_severity_score,
)

MAX_LENGTH = 150

# Lazy global model singletons
_tokenizer = None
_model = None
_device = None


def get_default_device():
    """Returns CUDA if available, otherwise CPU."""
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def get_model_and_tokenizer():
    """Lazily loads the model and tokenizer upon first call."""
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        dev = get_default_device()
        _tokenizer, _model = load_toxicity_model(device=dev)
    return _tokenizer, _model


def _resolve_threshold(threshold, label):
    """Resolves threshold whether provided as float or dict."""
    if isinstance(threshold, dict):
        return float(threshold.get(label, OPTIMAL_THRESHOLDS.get(label, 0.5)))
    if threshold is not None:
        return float(threshold)
    return float(OPTIMAL_THRESHOLDS.get(label, 0.5))


def predict_toxicity(
    text: str,
    threshold=None,
    tokenizer=None,
    model=None,
    device=None,
):
    """
    Predict toxicity probabilities and category detections for one comment.
    Supports custom per-label thresholds or policy presets.
    """
    if tokenizer is None or model is None:
        tokenizer, model = get_model_and_tokenizer()

    if device is None:
        device = next(model.parameters()).device

    cleaned_text = clean_text(text)

    inputs = tokenizer(
        cleaned_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.sigmoid(outputs.logits)[0]

    results = {}
    for label, probability in zip(LABELS, probabilities):
        score = float(probability.item())
        label_thresh = _resolve_threshold(threshold, label)
        results[label] = {
            "probability": round(score, 4),
            "threshold": label_thresh,
            "prediction": score >= label_thresh,
        }

    severity_score, severity_level = calculate_severity_score(results)
    results["_summary"] = {
        "severity_score": severity_score,
        "severity_level": severity_level,
        "is_toxic": any(results[l]["prediction"] for l in LABELS),
    }

    return results


def predict_batch(
    texts,
    threshold=None,
    batch_size=32,
    tokenizer=None,
    model=None,
    device=None,
):
    """
    Predict toxicity probabilities for multiple comments in batched chunks.
    """
    if tokenizer is None or model is None:
        tokenizer, model = get_model_and_tokenizer()

    if device is None:
        device = next(model.parameters()).device

    all_results = []
    cleaned_texts = [clean_text(text) for text in texts]

    for start in range(0, len(cleaned_texts), batch_size):
        batch_texts = cleaned_texts[start:start + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        probabilities = torch.sigmoid(outputs.logits)

        for prob_row in probabilities:
            row_result = {}
            for label, prob in zip(LABELS, prob_row):
                score = float(prob.item())
                label_thresh = _resolve_threshold(threshold, label)
                row_result[label] = {
                    "probability": round(score, 4),
                    "threshold": label_thresh,
                    "prediction": score >= label_thresh,
                }
            sev_score, sev_lvl = calculate_severity_score(row_result)
            row_result["_summary"] = {
                "severity_score": sev_score,
                "severity_level": sev_lvl,
                "is_toxic": any(row_result[l]["prediction"] for l in LABELS),
            }
            all_results.append(row_result)

    return all_results


def predict_batch_binary(
    texts,
    threshold=None,
    batch_size=32,
    tokenizer=None,
    model=None,
    device=None,
):
    """
    Fast batch prediction returning binary 0/1 indicator matrix.
    Supports per-label thresholds or fixed threshold.
    """
    if tokenizer is None or model is None:
        tokenizer, model = get_model_and_tokenizer()

    if device is None:
        device = next(model.parameters()).device

    all_predictions = []
    cleaned_texts = [clean_text(text) for text in texts]

    # Resolve threshold tensor [6]
    thresh_list = [_resolve_threshold(threshold, lbl) for lbl in LABELS]
    thresh_tensor = torch.tensor(thresh_list, device=device)

    for start in range(0, len(cleaned_texts), batch_size):
        batch_texts = cleaned_texts[start:start + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model(**inputs)
            probabilities = torch.sigmoid(outputs.logits)
            predictions = (probabilities >= thresh_tensor).int()

        all_predictions.extend(predictions.cpu().tolist())

    return all_predictions