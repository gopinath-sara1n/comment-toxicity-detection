import torch

from src.load_model import load_toxicity_model
from src.preprocessing import clean_text


LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

MAX_LENGTH = 150

tokenizer, model = load_toxicity_model()


def predict_toxicity(text, threshold=0.5):
    """Predict toxicity probabilities for one comment."""

    cleaned_text = clean_text(text)

    inputs = tokenizer(
        cleaned_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.sigmoid(outputs.logits)[0]

    results = {}

    for label, probability in zip(LABELS, probabilities):
        score = probability.item()

        results[label] = {
            "probability": round(score, 4),
            "prediction": score >= threshold,
        }

    return results


def predict_batch(texts, threshold=0.5, batch_size=32):
    """Predict toxicity probabilities for multiple comments."""

    all_results = []

    # Clean all comments using training preprocessing
    cleaned_texts = [
        clean_text(text)
        for text in texts
    ]

    for start in range(0, len(cleaned_texts), batch_size):

        batch_texts = cleaned_texts[
            start:start + batch_size
        ]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
        )

        with torch.no_grad():
            outputs = model(**inputs)

        probabilities = torch.sigmoid(outputs.logits)

        for probability_row in probabilities:

            row_result = {}

            for label, probability in zip(
                LABELS,
                probability_row,
            ):
                score = probability.item()

                row_result[label] = {
                    "probability": round(score, 4),
                    "prediction": score >= threshold,
                }

            all_results.append(row_result)

    return all_results



def predict_batch_binary(texts, threshold=0.5, batch_size=32):
    """
    Fast batch prediction for CSV uploads.

    Returns only 0/1 predictions for each toxicity label.
    """

    all_predictions = []

    # Same preprocessing used during model training
    cleaned_texts = [
        clean_text(text)
        for text in texts
    ]

    for start in range(0, len(cleaned_texts), batch_size):

        batch_texts = cleaned_texts[
            start:start + batch_size
        ]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
        )

        with torch.inference_mode():

            outputs = model(**inputs)

            probabilities = torch.sigmoid(
                outputs.logits
            )

            predictions = (
                probabilities >= threshold
            ).int()

        all_predictions.extend(
            predictions.cpu().tolist()
        )

    return all_predictions