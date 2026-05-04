"""
predict.py — Interface d'inférence unifiée.

Ce module est le point d'entrée pour faire des prédictions sur de
nouvelles critiques. Il gère automatiquement les deux types de modèles :
- Mode 'count' : régression logistique sur n-grammes (vocab = liste de tuples)
- Mode 'tfidf' : régression logistique sur TF-IDF (vocab = TfidfVectorizer)

Utilisé par :
- L'app Streamlit (app/streamlit_app.py)
- L'API FastAPI (api/main.py)
"""
import pickle
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer

from nlp_sentiment.config import DEVICE, MODEL_PATH, REVIEW_CLASSES, VOCAB_PATH
from nlp_sentiment.models import LogisticRegression
from nlp_sentiment.preprocessor import preprocess


# Cache global pour ne pas recharger le modèle à chaque prédiction
_MODEL_CACHE: dict = {}


def load_model_and_vocab(
    model_path: Path = MODEL_PATH,
    vocab_path: Path = VOCAB_PATH,
) -> tuple[LogisticRegression, object, str]:
    """
    Charge le modèle et le vocabulaire/vectorizer depuis le disque.

    Détecte automatiquement le format :
    - Si vocab_path contient un TfidfVectorizer → mode 'tfidf'
    - Sinon (liste de tuples) → mode 'count'

    Returns:
        Tuple (model, vocab_obj, mode) où :
        - model : LogisticRegression chargé en mode eval
        - vocab_obj : liste de tuples OU TfidfVectorizer
        - mode : 'count' ou 'tfidf'

    Raises:
        FileNotFoundError: Si modèle ou vocab introuvable.
                           L'utilisateur doit lancer train_champion_tfidf.py.
    """
    cache_key = (str(model_path), str(vocab_path))
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {model_path}\n"
            f"Lancez : python scripts/train_champion_tfidf.py"
        )
    if not vocab_path.exists():
        raise FileNotFoundError(
            f"Vocabulaire introuvable : {vocab_path}\n"
            f"Lancez : python scripts/train_champion_tfidf.py"
        )

    # Charge le vocab/vectorizer
    with open(vocab_path, "rb") as f:
        vocab_obj = pickle.load(f)

    # Détection automatique du mode
    if isinstance(vocab_obj, TfidfVectorizer):
        mode = "tfidf"
        input_dim = len(vocab_obj.vocabulary_)
    elif isinstance(vocab_obj, list):
        mode = "count"
        input_dim = len(vocab_obj)
    else:
        raise ValueError(
            f"Format de vocabulaire non reconnu : {type(vocab_obj).__name__}. "
            f"Attendu : list (mode count) ou TfidfVectorizer (mode tfidf)."
        )

    # Reconstruit et charge le modèle
    model = LogisticRegression(input_dim=input_dim, output_dim=len(REVIEW_CLASSES))
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()
    model = model.to(DEVICE)

    _MODEL_CACHE[cache_key] = (model, vocab_obj, mode)
    return model, vocab_obj, mode


def _build_feature_count(text: str, vocab: list[tuple]) -> torch.Tensor:
    """Construit un vecteur de comptage de bigrammes (mode count)."""
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
    """Construit un vecteur TF-IDF (mode tfidf)."""
    tokens = preprocess(text)
    text_joined = " ".join(tokens)
    sparse_features = vectorizer.transform([text_joined])
    feature = torch.tensor(
        sparse_features.toarray()[0],
        dtype=torch.float32,
        device=DEVICE,
    )
    return feature.unsqueeze(0)


def predict(text: str) -> dict:
    """
    Prédit le sentiment d'une critique de film.

    Args:
        text: Critique brute (texte en anglais).

    Returns:
        Dict avec :
        - 'label' : 'positif' ou 'négatif'
        - 'confidence' : probabilité de la classe prédite
        - 'probabilities' : dict avec les proba pour chaque classe
        - 'model_type' : 'count' ou 'tfidf'

    Example:
        >>> result = predict("This movie was absolutely wonderful!")
        >>> result['label']
        'positif'
    """
    if not text or not text.strip():
        raise ValueError("Le texte d'entrée ne peut pas être vide.")

    model, vocab_obj, mode = load_model_and_vocab()

    # Construction du vecteur d'entrée selon le mode
    if mode == "count":
        feature = _build_feature_count(text, vocab_obj)
    else:  # mode == "tfidf"
        feature = _build_feature_tfidf(text, vocab_obj)

    # Inférence
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