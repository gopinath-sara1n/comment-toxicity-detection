import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


MODEL_NAME = "gopinath-sara1n/comment-toxicity-distilbert"


def load_toxicity_model(device=None):
    """
    Load the fine-tuned DistilBERT toxicity model
    and tokenizer from Hugging Face Hub, mapped to target device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    return tokenizer, model