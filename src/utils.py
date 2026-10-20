import torch
import numpy as np


LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

LABEL_NAMES = {
    "toxic": "Toxic",
    "severe_toxic": "Severe Toxic",
    "obscene": "Obscene",
    "threat": "Threat",
    "insult": "Insult",
    "identity_hate": "Identity Hate",
}

# Optimal thresholds tuned on imbalanced validation sets to maximize per-class F1
OPTIMAL_THRESHOLDS = {
    "toxic": 0.45,
    "severe_toxic": 0.25,
    "obscene": 0.45,
    "threat": 0.20,
    "insult": 0.40,
    "identity_hate": 0.25,
}

POLICY_PRESETS = {
    "Balanced (F1-Optimized)": OPTIMAL_THRESHOLDS,
    "Strict Moderation (High Safety)": {
        "toxic": 0.30,
        "severe_toxic": 0.15,
        "obscene": 0.30,
        "threat": 0.10,
        "insult": 0.25,
        "identity_hate": 0.15,
    },
    "Relaxed (High Precision)": {
        "toxic": 0.60,
        "severe_toxic": 0.40,
        "obscene": 0.60,
        "threat": 0.35,
        "insult": 0.55,
        "identity_hate": 0.40,
    },
    "Standard (0.50 Baseline)": {
        "toxic": 0.50,
        "severe_toxic": 0.50,
        "obscene": 0.50,
        "threat": 0.50,
        "insult": 0.50,
        "identity_hate": 0.50,
    },
}

# Weights used for multi-hazard severity scoring
SEVERITY_WEIGHTS = {
    "threat": 2.0,
    "severe_toxic": 2.0,
    "identity_hate": 1.5,
    "toxic": 1.0,
    "insult": 1.0,
    "obscene": 0.8,
}


def calculate_severity_score(probabilities: dict) -> tuple:
    """
    Computes a composite Toxicity Severity Index (0 - 100) and severity level
    based on individual category probabilities and hazard weighting.
    """
    if not probabilities:
        return 0, "Safe"

    raw_probs = {
        label: float(probabilities.get(label, {}).get("probability", 0.0)
                     if isinstance(probabilities.get(label), dict)
                     else probabilities.get(label, 0.0))
        for label in LABELS
    }

    max_prob = max(raw_probs.values()) if raw_probs else 0.0
    total_weight = sum(SEVERITY_WEIGHTS.values())
    weighted_sum = sum(raw_probs[k] * SEVERITY_WEIGHTS[k] for k in LABELS) / total_weight

    # Peak-sensitive composite formula
    composite = max(max_prob, 0.4 * max_prob + 0.6 * (weighted_sum * 1.4))
    score = int(round(min(1.0, composite) * 100))

    if score <= 20:
        level = "Safe / Clean"
    elif score <= 45:
        level = "Low Risk"
    elif score <= 70:
        level = "Moderate Risk"
    elif score <= 85:
        level = "High Toxicity"
    else:
        level = "Critical / Severe Risk"

    return score, level


def explain_text_attribution(
    text: str,
    tokenizer,
    model,
    device=None,
    target_label_idx: int = 0
) -> tuple:
    """
    Computes gradient-based token attribution (saliency) for explainable AI.
    Formats words using native Streamlit Markdown color syntax (:red[...] and :orange[...])
    without any HTML or CSS tags.
    """
    if not text or not text.strip():
        return text, []

    if device is None:
        device = next(model.parameters()).device

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=150
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_ids = inputs["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    emb_layer = model.get_input_embeddings()
    embeddings = emb_layer(input_ids).detach().requires_grad_(True)

    outputs = model(inputs_embeds=embeddings, attention_mask=inputs["attention_mask"])
    logit = outputs.logits[0, target_label_idx]
    logit.backward()

    grads = embeddings.grad[0]
    scores = torch.norm(grads, dim=1).detach().cpu().numpy()

    # Filter out special tokens
    valid_tokens = []
    valid_scores = []
    for t, s in zip(tokens, scores):
        if t in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
        valid_tokens.append(t)
        valid_scores.append(float(s))

    if not valid_scores:
        return text, []

    # Min-max normalization
    scores_arr = np.array(valid_scores)
    s_min, s_max = scores_arr.min(), scores_arr.max()
    if s_max > s_min:
        norm_scores = (scores_arr - s_min) / (s_max - s_min)
    else:
        norm_scores = np.zeros_like(scores_arr)

    # Reassemble subwords (##) into words
    words = []
    word_scores = []
    for t, s in zip(valid_tokens, norm_scores):
        if t.startswith("##"):
            if words:
                words[-1] += t[2:]
                word_scores[-1] = max(word_scores[-1], float(s))
        else:
            words.append(t)
            word_scores.append(float(s))

    # Format using native Streamlit Markdown colored text
    formatted = []
    for w, s in zip(words, word_scores):
        if s >= 0.65:
            formatted.append(f":red[**{w}**]")
        elif s >= 0.40:
            formatted.append(f":orange[**{w}**]")
        else:
            formatted.append(w)

    highlighted_text = " ".join(formatted)
    word_details = [
        {"Word": w, "Attribution": round(float(s), 3)}
        for w, s in zip(words, word_scores)
    ]

    return highlighted_text, word_details
