"""Local-mode prediction interface for the deployed TF-IDF model.

Auto-detects whether the saved vocabulary is a TfidfVectorizer (TF-IDF mode)
or a list of n-gram tuples (count mode), and dispatches accordingly.

Used by the Streamlit app in offline / local mode. The HF Spaces deployment
loads models from the Hub via :mod:`predict_multi`.
"""
import pickle
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer

from nlp_sentiment.config import DEVICE, MODEL_PATH, REVIEW_CLASSES, VOCAB_PATH
from nlp_sentiment.models import LogisticRegression
from nlp_sentiment.preprocessor import preprocess


# Cache pour eviter de recharger le modele a chaque appel
_MODEL_CACHE: dict = {}


def load_model_and_vocab(
    model_path: Path = MODEL_PATH,
    vocab_path: Path = VOCAB_PATH,
) -> tuple[LogisticRegression, object, str]:
    """Load the trained model and its vocabulary from disk.

    Returns:
        Tuple (model, vocab_obj, mode) where mode is 'count' or 'tfidf'.

    Raises:
        FileNotFoundError: When the artifacts are missing on disk. Run
            ``scripts/train_champion_tfidf.py`` to generate them.
    """
    cache_key = (str(model_path), str(vocab_path))
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            f"Run: python scripts/train_champion_tfidf.py"
        )
    if not vocab_path.exists():
        raise FileNotFoundError(
            f"Vocab not found at {vocab_path}. "
            f"Run: python scripts/train_champion_tfidf.py"
        )

    with open(vocab_path, "rb") as f:
        vocab_obj = pickle.load(f)

    # Detection automatique du format (TfidfVectorizer vs liste de n-grammes)
    if isinstance(vocab_obj, TfidfVectorizer):
        mode = "tfidf"
        input_dim = len(vocab_obj.vocabulary_)
    elif isinstance(vocab_obj, list):
        mode = "count"
        input_dim = len(vocab_obj)
    else:
        raise ValueError(
            f"Unrecognized vocab format: {type(vocab_obj).__name__}. "
            f"Expected list (count mode) or TfidfVectorizer (tfidf mode)."
        )

    model = LogisticRegression(input_dim=input_dim, output_dim=len(REVIEW_CLASSES))
    model.load_state_dict(
        torch.load(model_path, map_location=DEVICE, weights_only=True)
    )
    model.eval()
    model = model.to(DEVICE)

    _MODEL_CACHE[cache_key] = (model, vocab_obj, mode)
    return model, vocab_obj, mode


def _build_feature_count(text: str, vocab: list[tuple]) -> torch.Tensor:
    """Build a count vector of bigrams (count mode)."""
    n = 2
    tokens = preprocess(text)
    ngram_to_idx = {ngram: i for i, ngram in enumerate(vocab)}

    feature = torch.zeros(len(vocab), device=DEVICE)
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i : i + n])
        if ngram in ngram_to_idx:
            feature[ngram_to_idx[ngram]] += 1
    return feature.unsqueeze(0)


def _build_feature_tfidf(text: str, vectorizer: TfidfVectorizer) -> torch.Tensor:
    """Build a TF-IDF vector (tfidf mode)."""
    tokens = preprocess(text)
    text_joined = " ".join(tokens)
    sparse = vectorizer.transform([text_joined])
    feature = torch.tensor(
        sparse.toarray()[0],
        dtype=torch.float32,
        device=DEVICE,
    )
    return feature.unsqueeze(0)


def predict(text: str) -> dict:
    """Predict the sentiment of a movie review.

    Args:
        text: Raw review text in English.

    Returns:
        Dict with keys 'label', 'confidence', 'probabilities', 'model_type'.

    Raises:
        ValueError: When the input text is empty.
    """
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty.")

    model, vocab_obj, mode = load_model_and_vocab()

    if mode == "count":
        feature = _build_feature_count(text, vocab_obj)
    else:
        feature = _build_feature_tfidf(text, vocab_obj)

    with torch.no_grad():
        logits = model(feature)
        probabilities = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()

    pred_idx = int(torch.tensor(probabilities).argmax().item())
    pred_class = REVIEW_CLASSES[pred_idx]
    label = "positif" if pred_class == "pos" else "négatif"

    return {
        "label": label,
        "confidence": round(probabilities[pred_idx], 4),
        "probabilities": {
            "négatif": round(probabilities[0], 4),
            "positif": round(probabilities[1], 4),
        },
        "model_type": mode,
    }