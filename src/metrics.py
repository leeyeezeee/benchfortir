import sys
import os
sys.path.append(os.getcwd())

import re
import string
from typing import Dict, Any, List, Union, Optional, Tuple, Set
from collections import Counter

from .math_equivalence import is_equiv


def normalize_answer(text: str, remove_articles: bool = False, remove_punctuations: bool = False) -> str:
    """Normalize answer text.

    NOTE
    ----
    This helper is used by both QA and Math evaluation. For QA-style F1 (SQuAD-like),
    we typically remove articles and punctuations.

    Args:
        text: The original answer text.
        remove_articles: Whether to remove articles (a/an/the).
        remove_punctuations: Whether to remove punctuation marks.

    Returns:
        The normalized answer.
    """
    if not isinstance(text, str):
        text = str(text)

    text = text.lower().strip()

    if remove_articles:
        text = re.sub(r"\b(a|an|the)\b", " ", text)

    if remove_punctuations:
        exclude = set(string.punctuation)
        text = "".join(ch for ch in text if ch not in exclude)

    return " ".join(text.split())


def compute_token_overlap(prediction: str, reference: str) -> Tuple[int, int, int]:
    """Compute token overlap between prediction and reference."""
    prediction_tokens = prediction.split()
    reference_tokens = reference.split()

    common = Counter(prediction_tokens) & Counter(reference_tokens)
    num_same = sum(common.values())

    return num_same, len(prediction_tokens), len(reference_tokens)


def compute_f1_score(num_same: int, pred_len: int, ref_len: int) -> float:
    """Compute token-level F1 score."""
    if num_same == 0:
        return 0.0

    precision = num_same / pred_len if pred_len > 0 else 0
    recall = num_same / ref_len if ref_len > 0 else 0

    return (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0


def evaluate_math_prediction(prediction: str, reference: str) -> Dict[str, Union[int, float]]:
    """Evaluate mathematical prediction result."""
    normalized_prediction = normalize_answer(prediction)
    normalized_reference = normalize_answer(reference)

    em = int(normalized_prediction == normalized_reference)
    acc = int(normalized_reference in normalized_prediction)

    num_same, pred_len, ref_len = compute_token_overlap(normalized_prediction, normalized_reference)
    f1 = compute_f1_score(num_same, pred_len, ref_len)

    math_equal = int(is_equiv(normalized_prediction, normalized_reference))

    return {"em": em, "acc": acc, "f1": f1, "math_equal": math_equal}


def evaluate_qa_prediction(prediction: str, references: Union[str, List[str], bool]) -> Dict[str, Union[int, float]]:
    """Evaluate QA prediction result.

    This uses SQuAD-style normalization by default:
    - lowercasing
    - removing articles (a/an/the)
    - removing punctuations

    The metric for a sample is computed against each reference and the maximum
    score is taken (standard practice for multi-reference QA).

    Args:
        prediction: Predicted answer string.
        references: One reference answer or a list of reference answers.

    Returns:
        Dict[str, Union[int, float]] with keys: em, acc, f1, math_equal.
    """
    # Ensure references is a list
    if isinstance(references, str):
        if not references.startswith("["):
            references = [references]
        else:
            references = [e.strip() for e in re.split(r",\s*", references.strip("[]"))]

    # Handle boolean answers
    if isinstance(references, list) and all(isinstance(r, bool) for r in references):
        references = [str(r) for r in references]
    elif isinstance(references, bool):
        references = [str(references)]

    result: Dict[str, Union[int, float]] = {"em": 0, "acc": 0, "f1": 0.0, "math_equal": 0}

    # IMPORTANT: remove punctuation for QA to align with SQuAD-style F1
    normalized_prediction = normalize_answer(prediction, remove_articles=True, remove_punctuations=True)

    try:
        for reference in references:  # type: ignore[assignment]
            normalized_reference = normalize_answer(reference, remove_articles=True, remove_punctuations=True)

            em = int(normalized_prediction == normalized_reference)
            acc = int(normalized_reference in normalized_prediction)

            num_same, pred_len, ref_len = compute_token_overlap(normalized_prediction, normalized_reference)
            f1 = compute_f1_score(num_same, pred_len, ref_len)

            result["em"] = max(int(result["em"]), em)
            result["acc"] = max(int(result["acc"]), acc)
            result["f1"] = max(float(result["f1"]), float(f1))

            # Keep math_equal for compatibility (sometimes QA answers are numeric)
            math_equal = int(is_equiv(normalized_prediction, normalized_reference))
            result["math_equal"] = max(int(result["math_equal"]), math_equal)

    except Exception as e:
        print(f"Error in evaluate_qa_prediction: {str(e)}")
        print(f"Prediction: {prediction}")
        print(f"References: {references}")
        raise

    return result
