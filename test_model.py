import sys
from src.preprocessing import clean_text
from src.utils import (
    LABELS,
    LABEL_NAMES,
    OPTIMAL_THRESHOLDS,
    calculate_severity_score,
)
from src.prediction import (
    predict_toxicity,
    predict_batch_binary,
)


def test_labels_configuration():
    assert len(LABELS) == 6
    for label in LABELS:
        assert label in LABEL_NAMES
        assert label in OPTIMAL_THRESHOLDS


def test_clean_text_normalizes_string():
    raw = "Check this   link http://example.com/test! It's awesome!!!"
    cleaned = clean_text(raw)
    assert isinstance(cleaned, str)
    assert "http" not in cleaned
    assert len(cleaned) > 0


def test_severity_calculation_safe():
    safe_probs = {l: 0.05 for l in LABELS}
    score, level = calculate_severity_score(safe_probs)
    assert 0 <= score <= 20
    assert "Safe" in level


def test_severity_calculation_high_threat():
    threat_probs = {l: 0.02 for l in LABELS}
    threat_probs["threat"] = 0.95
    score, level = calculate_severity_score(threat_probs)
    assert score >= 75
    assert "Severe" in level or "High" in level


def test_predict_toxicity_live():
    res = predict_toxicity("Thank you so much for your kind help and support.")
    assert "_summary" in res
    assert res["_summary"]["severity_score"] < 40

    for l in LABELS:
        assert l in res
        assert 0.0 <= res[l]["probability"] <= 1.0
        assert isinstance(res[l]["prediction"], (bool, int))


def test_predict_toxicity_offensive():
    res = predict_toxicity("You are a disgusting idiot, get out of here.")
    assert "_summary" in res
    # Either toxic or insult should have elevated probability
    assert res["toxic"]["probability"] > 0.4 or res["insult"]["probability"] > 0.4


def test_predict_batch_binary():
    texts = [
        "Have a wonderful weekend ahead!",
        "I will kill you and destroy your life.",
    ]
    preds = predict_batch_binary(texts, batch_size=2)
    assert len(preds) == 2
    assert len(preds[0]) == 6
    assert len(preds[1]) == 6
    for row in preds:
        for val in row:
            assert val in (0, 1)


if __name__ == "__main__":
    try:
        import pytest
        sys.exit(pytest.main(["-v", __file__]))
    except ImportError:
        print("pytest not installed; running test runner directly...")
        test_fns = [
            test_labels_configuration,
            test_clean_text_normalizes_string,
            test_severity_calculation_safe,
            test_severity_calculation_high_threat,
            test_predict_toxicity_live,
            test_predict_toxicity_offensive,
            test_predict_batch_binary,
        ]
        for fn in test_fns:
            print(f"Running {fn.__name__}...", end=" ")
            fn()
            print("PASSED")
        print("\nAll tests passed successfully!")
