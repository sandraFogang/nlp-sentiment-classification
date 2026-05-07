"""Word-level interpretability across all three deployed paradigms.

For TF-IDF we use the closed-form coefficient contribution.
For BiLSTM and DistilBERT we use Leave-One-Out occlusion.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from nlp_sentiment.config import REVIEW_CLASSES
from nlp_sentiment.preprocessor import preprocess


MAX_OCCLUSION_TOKENS = 30


def explain_tfidf(text: str, model, vectorizer, top_k: int = 5):
    tokens = preprocess(text)
    joined = " ".join(tokens)
    tfidf_vec = vectorizer.transform([joined]).toarray()[0]

    weights = model.linear.weight.detach().cpu().numpy()
    pos_idx = REVIEW_CLASSES.index("pos")
    neg_idx = REVIEW_CLASSES.index("neg")

    contrib_diff = tfidf_vec * (weights[pos_idx] - weights[neg_idx])
    nonzero = tfidf_vec > 0
    contrib_diff = np.where(nonzero, contrib_diff, 0.0)

    if not nonzero.any():
        return [], []

    feature_names = np.array(vectorizer.get_feature_names_out())

    pos_indices = np.argsort(contrib_diff)[::-1][:top_k]
    positive_words = [
        {"word": feature_names[i], "contribution": float(contrib_diff[i])}
        for i in pos_indices
        if contrib_diff[i] > 0
    ]

    neg_indices = np.argsort(contrib_diff)[:top_k]
    negative_words = [
        {"word": feature_names[i], "contribution": float(-contrib_diff[i])}
        for i in neg_indices
        if contrib_diff[i] < 0
    ]

    return positive_words, negative_words


def explain_by_occlusion(
    text: str,
    predict_fn: Callable[[str], dict],
    top_k: int = 5,
    max_tokens: int = MAX_OCCLUSION_TOKENS,
):
    tokens = preprocess(text)[:max_tokens]
    if not tokens:
        return [], []

    baseline = predict_fn(text)
    baseline_pos = baseline["probabilities"]["positif"]

    contributions = []
    for i, token in enumerate(tokens):
        masked_tokens = tokens[:i] + tokens[i + 1:]
        masked_text = " ".join(masked_tokens) if masked_tokens else "the"
        masked_pred = predict_fn(masked_text)
        masked_pos = masked_pred["probabilities"]["positif"]

        delta = baseline_pos - masked_pos
        contributions.append({"word": token, "contribution": float(delta)})

    positive_words = sorted(
        [c for c in contributions if c["contribution"] > 0],
        key=lambda c: c["contribution"],
        reverse=True,
    )[:top_k]

    negative_words = [
        {"word": c["word"], "contribution": float(-c["contribution"])}
        for c in sorted(
            [c for c in contributions if c["contribution"] < 0],
            key=lambda c: c["contribution"],
        )[:top_k]
    ]

    return positive_words, negative_words