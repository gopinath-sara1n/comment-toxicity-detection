import re

try:
    import contractions
except ImportError:
    contractions = None

try:
    import emoji
except ImportError:
    emoji = None

try:
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
except ImportError:
    nltk = None
    WordNetLemmatizer = None
    word_tokenize = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None


# Download NLTK resources safely only if available and missing
def download_nltk_resources():
    if nltk is None:
        return
    resources = {
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
    }

    for resource_path, resource_name in resources.items():
        try:
            nltk.data.find(resource_path)
        except (LookupError, Exception):
            try:
                nltk.download(resource_name, quiet=True)
            except Exception:
                pass


download_nltk_resources()

lemmatizer = None
if WordNetLemmatizer is not None:
    try:
        lemmatizer = WordNetLemmatizer()
    except Exception:
        lemmatizer = None


def clean_text(text: str) -> str:
    """
    Clean text matching training preprocessing:
    - Lowercase
    - Remove URLs, emails, HTML, emojis
    - Expand contractions
    - Normalize repeated punctuation and extra whitespace
    - Tokenize and lemmatize
    """
    text = str(text).lower()

    # Fast check: Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # Remove HTML only if markup tags exist
    if "<" in text and ">" in text:
        if BeautifulSoup is not None:
            try:
                text = BeautifulSoup(text, "html.parser").get_text(" ")
            except Exception:
                text = re.sub(r"<[^>]+>", " ", text)
        else:
            text = re.sub(r"<[^>]+>", " ", text)

    # Remove emojis
    if emoji is not None:
        text = emoji.replace_emoji(text, replace=" ")
    else:
        text = re.sub(r"[\U00010000-\U0010ffff]", " ", text)

    # Expand contractions
    if contractions is not None:
        try:
            text = contractions.fix(text)
        except Exception:
            pass

    # Keep only alphabets, ! and ?
    text = re.sub(r"[^a-z!?\s]", " ", text)

    # Normalize repeated ! and ?
    text = re.sub(r"!+", " ! ", text)
    text = re.sub(r"\?+", " ? ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize and lemmatize
    if word_tokenize is not None:
        try:
            words = word_tokenize(text)
            if lemmatizer is not None:
                words = [lemmatizer.lemmatize(word) for word in words]
            return " ".join(words)
        except Exception:
            pass

    return text