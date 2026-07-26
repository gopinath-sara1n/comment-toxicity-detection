from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


MODEL_NAME = "gopinath-sara1n/comment-toxicity-distilbert"


def load_toxicity_model():
    """
    Load the fine-tuned DistilBERT toxicity model
    and tokenizer from Hugging Face Hub.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME
    )

    model.eval()

    return tokenizer, model