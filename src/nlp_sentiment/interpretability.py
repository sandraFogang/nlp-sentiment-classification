"""Word-level interpretability across all three deployed paradigms.

For TF-IDF we use the closed-form coefficient contribution (linear model,
exact and instant). For BiLSTM and DistilBERT we use Leave-One-Out occlusion:
each token is masked individually, and the resulting drop in positive
probability is interpreted as that token's contribution.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from nlp_sentiment.config import REVIEW_CLASSES
from nlp_sentiment.preprocessor import preprocess


# Borne sur le nombre de tokens occlus pour limiter le temps de calcul
# (BERT prend ~30s pour 30 tokens, ~1min pour 60)
MAX_OCCLUSION_TOKENS = 30


def explain_tfidf(text: str, model, vectorizer, top_k: int = 5):
    """Return the top-k positive and negative contributing words for a TF-IDF model.

    Contribution of word w = ``tfidf(w) * (weight_pos - weight_neg)``.

    Args:
        text: Raw review text.
        model: Trained ``LogisticRegression`` instance with ``.linear`` layer.
        vectorizer: Fitted ``TfidfVectorizer``.
        top_k: Number of words to return per polarity.

    Returns:
        Tuple ``(positive_words, negative_words)``. Each is a list of
        ``{"word": str, "contribution": float}`` dicts.
    """
    tokens = preprocess(text)
    joined = " ".join(tokens)
    tfidf_vec = vectorizer.transform([joined]).toarray()[0]

    weights = model.linear.weight.detach().cpu().numpy()
    pos_idx = REVIEW_CLASSES.index("pos")
    neg_idx = REVIEW_CLASSES.index("neg")

    # Difference des poids = direction de la contribution
    contrib_diff = tfidf_vec * (weights[pos_idx] - weights[neg_idx])

    # On ne garde que les mots presents dans le texte (TF-IDF non nul)
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
    """Return the top-k contributing words via Leave-One-Out occlusion.

    For each token in the input, we remove it, re-run the prediction, and
    measure the change in positive probability. A positive delta means the
    token was pushing toward the positive class.

    Args:
        text: Raw review text.
        predict_fn: Callable taking text and returning a prediction dict.
        top_k: Number of words to return per polarity.
        max_tokens: Maximum number of tokens to occlude (caps inference time).

    Returns:
        Tuple ``(positive_words, negative_words)``.
    """
    tokens = preprocess(text)[:max_tokens]
    if not tokens:
        return [], []

    baseline_pos = predict_fn(text)["probabilities"]["positif"]

    contributions = []
    for i, token in enumerate(tokens):
        # On masque le token i en le retirant de la liste
        masked = tokens[:i] + tokens[i + 1:]
        masked_text = " ".join(masked) if masked else "the"
        masked_pos = predict_fn(masked_text)["probabilities"]["positif"]

        # delta positif = retirer ce mot baisse la proba positive => mot positif
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