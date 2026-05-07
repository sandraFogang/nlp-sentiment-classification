"""Model artifact loaders backed by the Hugging Face Hub.

Each loader fetches the trained weights and supporting files for one
of the three deployed paradigms. Files are cached locally by
``huggingface_hub`` after the first download.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download, snapshot_download


# Trois dépôts modèles distincts sur HF Hub (un par paradigme)
HF_REPOS = {
    "tfidf": "sandraFogang/nlp-sentiment-tfidf",
    "bilstm": "sandraFogang/nlp-sentiment-bilstm",
    "distilbert": "sandraFogang/nlp-sentiment-distilbert",
}


def _download(repo_id: str, filename: str) -> Path:
    """Download a single file from a HF Hub model repo."""
    return Path(hf_hub_download(repo_id=repo_id, filename=filename))


def load_tfidf_artifacts() -> tuple[Any, Any]:
    """Return (state_dict, fitted TfidfVectorizer) for the TF-IDF model."""
    weights_path = _download(HF_REPOS["tfidf"], "classifier.pt")
    vectorizer_path = _download(HF_REPOS["tfidf"], "vectorizer.pkl")

    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    return state_dict, vectorizer


def load_bilstm_artifacts() -> tuple[Any, Any]:
    """Return (state_dict, word2idx mapping) for the BiLSTM model."""
    weights_path = _download(HF_REPOS["bilstm"], "classifier.pt")
    vocab_path = _download(HF_REPOS["bilstm"], "vocab.pkl")

    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    return state_dict, vocab


def load_distilbert_artifacts() -> tuple[Any, Path]:
    """Return (state_dict, tokenizer_dir) for the DistilBERT model."""
    weights_path = _download(HF_REPOS["distilbert"], "classifier.pt")

    # Le tokenizer est un dossier (config + vocab) - on le télécharge en bloc
    tokenizer_dir = Path(snapshot_download(
        repo_id=HF_REPOS["distilbert"],
        allow_patterns=["tokenizer/*"],
    )) / "tokenizer"

    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    return state_dict, tokenizer_dir