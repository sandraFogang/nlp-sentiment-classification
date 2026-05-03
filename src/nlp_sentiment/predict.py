"""
predict.py — Interface d'inférence unifiée.

Ce module est le point d'entrée pour faire des prédictions sur de
nouvelles critiques. Il est utilisé par :
- L'app Streamlit (app/streamlit_app.py)
- L'API FastAPI (api/main.py)
- Les scripts batch (scripts/predict_batch.py)
"""
import pickle
from pathlib import Path

import torch
import torch.nn.functional as F

from nlp_sentiment.config import DEVICE, MODEL_PATH, REVIEW_CLASSES, VOCAB_PATH
from nlp_sentiment.models import LogisticRegression
from nlp_sentiment.preprocessor import preprocess


# Cache global : on ne charge le modèle qu'une seule fois en mémoire
# (sinon Streamlit le rechargerait à chaque clic, ce qui serait très lent)
_MODEL_CACHE: dict = {}


def load_model_and_vocab(
    model_path: Path = MODEL_PATH,
    vocab_path: Path = VOCAB_PATH,
) -> tuple[LogisticRegression, list[tuple]]:
    """
    Charge le modèle et le vocabulaire depuis le disque.

    Le modèle est mis en cache : les appels suivants sont instantanés.

    Args:
        model_path: Chemin vers le fichier .pt du modèle.
        vocab_path: Chemin vers le fichier .pkl du vocabulaire.

    Returns:
        Tuple (modèle chargé en mode eval, vocabulaire de bigrammes).

    Raises:
        FileNotFoundError: Si le modèle ou le vocabulaire n'existe pas.
                           L'utilisateur doit lancer `python scripts/train_model.py`.
    """
    cache_key = (str(model_path), str(vocab_path))
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {model_path}\n"
            f"Lancez d'abord : python scripts/train_model.py"
        )
    if not vocab_path.exists():
        raise FileNotFoundError(
            f"Vocabulaire introuvable : {vocab_path}\n"
            f"Lancez d'abord : python scripts/train_model.py"
        )

    # Charge le vocabulaire
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    # Reconstruit le modèle avec les mêmes dimensions
    model = LogisticRegression(input_dim=len(vocab), output_dim=len(REVIEW_CLASSES))
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()
    model = model.to(DEVICE)

    _MODEL_CACHE[cache_key] = (model, vocab)
    return model, vocab


def predict(text: str) -> dict:
    """
    Prédit le sentiment d'une critique de film.

    Cette fonction est le point d'entrée principal pour Streamlit et FastAPI.

    Args:
        text: Critique brute (texte en anglais).

    Returns:
        Dictionnaire avec :
        - 'label' : 'positif' ou 'négatif'
        - 'confidence' : probabilité de la classe prédite (entre 0 et 1)
        - 'probabilities' : dict avec les proba pour chaque classe

    Example:
        >>> result = predict("This movie was absolutely wonderful!")
        >>> result['label']
        'positif'
    """
    if not text or not text.strip():
        raise ValueError("Le texte d'entrée ne peut pas être vide.")

    model, vocab = load_model_and_vocab()
    n = 2  # bigrammes (modèle déployé)

    # Tokenisation
    tokens = preprocess(text)

    # Construction du vecteur de comptage des bigrammes
    # Le tenseur est créé directement sur DEVICE pour éviter un transfert
    ngram_to_idx = {ngram: i for i, ngram in enumerate(vocab)}
    feature = torch.zeros(len(vocab), device=DEVICE)
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i : i + n])
        if ngram in ngram_to_idx:
            feature[ngram_to_idx[ngram]] += 1

    # Ajoute la dimension batch (1, vocab_size) — toujours sur DEVICE
    feature = feature.unsqueeze(0)

    # Inférence
    with torch.no_grad():
        logits = model(feature)
        # On déplace sur CPU avant .tolist() (les listes Python vivent toujours sur CPU)
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
    }