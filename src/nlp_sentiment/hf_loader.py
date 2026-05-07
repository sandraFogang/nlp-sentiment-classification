"""Lazy model loading from the Hugging Face Hub."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download, snapshot_download


HF_REPOS = {
    "tfidf": "sandraFogang/nlp-sentiment-tfidf",
    "bilstm": "sandraFogang/nlp-sentiment-bilstm",
    "distilbert": "sandraFogang/nlp-sentiment-distilbert",
}


def _download(repo_id: str, filename: str) -> Path:
    return Path(hf_hub_download(repo_id=repo_id, filename=filename))


def load_tfidf_artifacts() -> tuple[Any, Any]:
    weights_path = _download(HF_REPOS["tfidf"], "classifier.pt")
    vectorizer_path = _download(HF_REPOS["tfidf"], "vectorizer.pkl")

    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    return state_dict, vectorizer


def load_bilstm_artifacts() -> tuple[Any, Any]:
    weights_path = _download(HF_REPOS["bilstm"], "classifier.pt")
    vocab_path = _download(HF_REPOS["bilstm"], "vocab.pkl")

    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    return state_dict, vocab


def load_distilbert_artifacts() -> tuple[Any, Path]:
    weights_path = _download(HF_REPOS["distilbert"], "classifier.pt")

    tokenizer_dir = Path(snapshot_download(
        repo_id=HF_REPOS["distilbert"],
        allow_patterns=["tokenizer/*"],
    )) / "tokenizer"

    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    return state_dict, tokenizer_dir