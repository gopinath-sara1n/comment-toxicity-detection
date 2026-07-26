import re

import contractions
import emoji
import nltk
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# Download NLTK resources only if they are missing.
def download_nltk_resources():
    resources = {
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
    }

    for resource_path, resource_name in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(resource_name, quiet=True)


download_nltk_resources()

lemmatizer = WordNetLemmatizer()


def clean_text(text):
    """Clean text using the same preprocessing used during training."""

    # Convert input to string
    text = str(text)

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # Remove HTML
    text = BeautifulSoup(text, "html.parser").get_text(" ")

    # Remove emojis
    text = emoji.replace_emoji(text, replace=" ")

    # Expand contractions
    text = contractions.fix(text)

    # Keep only alphabets, ! and ?
    text = re.sub(r"[^a-z!?\s]", " ", text)

    # Normalize repeated ! and ?
    text = re.sub(r"!+", " ! ", text)
    text = re.sub(r"\?+", " ? ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize
    words = word_tokenize(text)

    # Lemmatize
    words = [
        lemmatizer.lemmatize(word)
        for word in words
    ]

    return " ".join(words)