"""interpretability.py — Word-level contribution analysis for the TF-IDF model.

Returns the top words pushing the prediction toward 'positif' or 'négatif'.
Only works for linear models on TF-IDF features (logistic regression).
"""
from typing import Tuple

import numpy as np

from nlp_sentiment.config import REVIEW_CLASSES
from nlp_sentiment.predict import load_model_and_vocab
from nlp_sentiment.preprocessor import preprocess


def get_word_contributions(text: str, top_k: int = 5) -> Tuple[list, list]:
    """Compute the top-k words most contributing to positive and negative predictions.

    For a linear model on TF-IDF, the contribution of word w to class c equals
    ``tfidf_value(w) * logistic_weight(w, c)``. We return the top-k for both classes.

    Args:
        text: Raw review text.
        top_k: Number of words to return per class (default 5).

    Returns:
        Tuple of (positive_words, negative_words). Each is a list of dicts with
        keys 'word' and 'contribution'.

    Raises:
        ValueError: If the loaded model is not in TF-IDF mode.
    """
    model, vocab_obj, mode = load_model_and_vocab()

    if mode != "tfidf":
        raise ValueError(
            f"Word contributions are only available for the TF-IDF model "
            f"(current mode: {mode!r}). Other models would require LIME or SHAP."
        )

    # Vectoriser le texte avec le même pipeline que pendant l'entraînement
    tokens = preprocess(text)
    text_joined = " ".join(tokens)
    tfidf_vec = vocab_obj.transform([text_joined]).toarray()[0]

    # Récupérer les poids du classifieur (matrice (2, n_features))
    weights = model.linear.weight.detach().cpu().numpy()

    pos_idx = REVIEW_CLASSES.index("pos")
    neg_idx = REVIEW_CLASSES.index("neg")

    # Contribution = TF-IDF × poids
    contrib_pos = tfidf_vec * weights[pos_idx]
    contrib_neg = tfidf_vec * weights[neg_idx]

    # Différence : positif > 0 indique que le mot pousse vers 'positif'
    contrib_diff = contrib_pos - contrib_neg

    # Mots vraiment présents dans le texte (TF-IDF non-zéro)
    nonzero_mask = tfidf_vec > 0
    if nonzero_mask.sum() == 0:
        return [], []

    feature_names = np.array(vocab_obj.get_feature_names_out())

    # Masquer la contribution des mots absents (pour qu'ils soient ignorés au tri)
    contrib_diff_masked = np.where(nonzero_mask, contrib_diff, 0.0)

    # Top-k mots positifs
    pos_indices = np.argsort(contrib_diff_masked)[::-1][:top_k]
    positive_words = [
        {"word": feature_names[i], "contribution": float(contrib_diff_masked[i])}
        for i in pos_indices
        if contrib_diff_masked[i] > 0
    ]

    # Top-k mots négatifs (contribution la plus négative → on prend la valeur absolue)
    neg_indices = np.argsort(contrib_diff_masked)[:top_k]
    negative_words = [
        {"word": feature_names[i], "contribution": float(-contrib_diff_masked[i])}
        for i in neg_indices
        if contrib_diff_masked[i] < 0
    ]

    return positive_words, negative_words