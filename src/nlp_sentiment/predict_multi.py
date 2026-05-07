"""Unified prediction interface across the three deployed paradigms."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from nlp_sentiment.config import REVIEW_CLASSES
from nlp_sentiment.hf_loader import (
    load_bilstm_artifacts,
    load_distilbert_artifacts,
    load_tfidf_artifacts,
)
from nlp_sentiment.models import LogisticRegression, BiLSTMClassifierV2
from nlp_sentiment.bert_model import BertSentimentClassifier
from nlp_sentiment.preprocessor import preprocess


DEVICE = torch.device("cpu")


def _predict_with_model(model: torch.nn.Module, feature: torch.Tensor) -> dict:
    model.eval()
    with torch.no_grad():
        logits = model(feature)
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().tolist()

    pred_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
    pred_class = REVIEW_CLASSES[pred_idx]
    label = "positif" if pred_class == "pos" else "négatif"

    return {
        "label": label,
        "confidence": round(probs[pred_idx], 4),
        "probabilities": {
            "négatif": round(probs[0], 4),
            "positif": round(probs[1], 4),
        },
    }


def _build_tfidf_predictor(state_dict: dict, vectorizer: Any):
    input_dim = len(vectorizer.vocabulary_)
    model = LogisticRegression(input_dim=input_dim, output_dim=len(REVIEW_CLASSES))
    model.load_state_dict(state_dict)
    model.to(DEVICE)

    def predict(text: str) -> dict:
        tokens = preprocess(text)
        joined = " ".join(tokens)
        sparse = vectorizer.transform([joined]).toarray()
        feature = torch.tensor(sparse, dtype=torch.float32, device=DEVICE)
        return _predict_with_model(model, feature)

    return predict, model, vectorizer


def _build_bilstm_predictor(state_dict: dict, vocab: dict):
    word2idx = vocab["word2idx"] if isinstance(vocab, dict) else vocab
    pad_idx = word2idx.get("<PAD>", 0)
    unk_idx = word2idx.get("<UNK>", 1)
    vocab_size = len(word2idx)

    model = BiLSTMClassifierV2(
        vocab_size=vocab_size,
        embedding_dim=300,
        hidden_dim=128,
        num_layers=2,
        num_classes=len(REVIEW_CLASSES),
        pad_idx=pad_idx,
    )
    model.load_state_dict(state_dict)
    model.to(DEVICE)

    def text_to_ids(text: str, max_len: int = 200) -> torch.Tensor:
        tokens = preprocess(text)[:max_len]
        ids = [word2idx.get(tok, unk_idx) for tok in tokens] or [pad_idx]
        if len(ids) < max_len:
            ids = ids + [pad_idx] * (max_len - len(ids))
        return torch.tensor([ids], dtype=torch.long, device=DEVICE)

    def predict(text: str) -> dict:
        feature = text_to_ids(text)
        return _predict_with_model(model, feature)

    return predict, model, word2idx


def _build_distilbert_predictor(state_dict: dict, tokenizer_dir):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    model = BertSentimentClassifier(
        model_name="distilbert-base-uncased",
        freeze_bert=False,
        dropout_rate=0.3,
    )
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    def predict(text: str) -> dict:
        encoded = tokenizer(
            text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(DEVICE)
        attention_mask = encoded["attention_mask"].to(DEVICE)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().tolist()

        pred_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
        pred_class = REVIEW_CLASSES[pred_idx]
        label = "positif" if pred_class == "pos" else "négatif"

        return {
            "label": label,
            "confidence": round(probs[pred_idx], 4),
            "probabilities": {
                "négatif": round(probs[0], 4),
                "positif": round(probs[1], 4),
            },
        }

    return predict, model, tokenizer


BUILDERS = {
    "tfidf": (load_tfidf_artifacts, _build_tfidf_predictor),
    "bilstm": (load_bilstm_artifacts, _build_bilstm_predictor),
    "distilbert": (load_distilbert_artifacts, _build_distilbert_predictor),
}


def build_predictor(model_key: str):
    if model_key not in BUILDERS:
        raise ValueError(f"Unknown model: {model_key}")
    loader, builder = BUILDERS[model_key]
    artifacts = loader()
    return builder(*artifacts)