"""
train_lstm.py — Entraînement du LSTM bidirectionnel avec GloVe.

Usage local (CPU, lent) :
    python scripts/train_lstm.py

Usage Colab (GPU, rapide) :
    Voir notebooks/lstm_colab.ipynb (à venir).

Hyperparamètres principaux :
- Vocabulaire : 30 000 mots, min_count=5
- Séquence : 512 mots (padding/troncature)
- Embeddings : GloVe 100d (avec fine-tuning)
- LSTM : bidirectionnel, hidden 64, dropout 0.3
- Optimisation : Adam, lr=0.001, batch 32
"""
import json
import pickle
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nlp_sentiment.config import (
    BATCH_SIZE,
    DEVICE,
    EARLY_STOPPING_MIN_DELTA,
    EARLY_STOPPING_PATIENCE,
    EXPERIMENTS_PATH,
    LEARNING_RATE,
    MODELS_DIR,
    OUTPUTS_DIR,
    RANDOM_SEED,
    REVIEW_CLASSES,
    TORCH_SEED,
)
from nlp_sentiment.data import describe_splits, load_imdb_splits
from nlp_sentiment.evaluate import compute_metrics, predict_on_dataloader
from nlp_sentiment.glove_utils import build_embedding_matrix, load_glove_embeddings
from nlp_sentiment.lstm_data import prepare_lstm_data
from nlp_sentiment.models import BiLSTMClassifier
from nlp_sentiment.train import train


# ============================================================================
# Configuration
# ============================================================================
EXPERIMENT_NAME = "bilstm_glove_baseline"

# Données
MAX_VOCAB_SIZE = 30000
MIN_COUNT = 5
MAX_SEQ_LEN = 512

# GloVe
GLOVE_PATH = PROJECT_ROOT / "data" / "glove" / "glove.6B.100d.txt"
EMBEDDING_DIM = 100

# Modèle
HIDDEN_DIM = 64
DROPOUT_RATE = 0.3

# Entraînement
LSTM_LEARNING_RATE = 0.001
LSTM_BATCH_SIZE = 64  # plus gros que pour les n-grammes (les séquences sont compactes en RAM)
LSTM_MAX_EPOCHS = 15

# Production
SAVE_AS_PRODUCTION = False  # à True UNIQUEMENT si on adopte ce modèle


# ============================================================================
# Logging des expériences (réutilise le format existant)
# ============================================================================
def log_experiment(experiment: dict) -> None:
    EXPERIMENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if EXPERIMENTS_PATH.exists():
        with open(EXPERIMENTS_PATH, "r", encoding="utf-8") as f:
            history_log = json.load(f)
    else:
        history_log = []
    history_log.append(experiment)
    with open(EXPERIMENTS_PATH, "w", encoding="utf-8") as f:
        json.dump(history_log, f, indent=2, ensure_ascii=False)
    print(f"\nExpérience enregistrée : {EXPERIMENTS_PATH}")


# ============================================================================
# Pipeline principal
# ============================================================================
def main() -> None:
    random.seed(RANDOM_SEED)
    torch.manual_seed(TORCH_SEED)

    print("=" * 60)
    print(f"ENTRAÎNEMENT LSTM : {EXPERIMENT_NAME}")
    print(f"Device : {DEVICE}")
    print("=" * 60)

    # === 1. Chargement IMDB ===
    print("\n--- ÉTAPE 1 : Chargement IMDB ---")
    train_data, val_data, test_data = load_imdb_splits()
    describe_splits(train_data, val_data, test_data)

    # === 2. Préparation des séquences ===
    print("\n--- ÉTAPE 2 : Préparation des séquences ---")
    train_dataset, val_dataset, vocab = prepare_lstm_data(
        train_data,
        val_data,
        max_vocab_size=MAX_VOCAB_SIZE,
        min_count=MIN_COUNT,
        max_seq_len=MAX_SEQ_LEN,
    )

    train_loader = DataLoader(train_dataset, batch_size=LSTM_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=LSTM_BATCH_SIZE, shuffle=False)

    # === 3. Chargement de GloVe ===
    print("\n--- ÉTAPE 3 : Chargement de GloVe ---")
    glove_embeddings = load_glove_embeddings(GLOVE_PATH, EMBEDDING_DIM)
    embedding_matrix = build_embedding_matrix(vocab, glove_embeddings, EMBEDDING_DIM)

    # === 4. Modèle ===
    print("\n--- ÉTAPE 4 : Construction du modèle BiLSTM ---")
    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        emb_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=len(REVIEW_CLASSES),
        dropout_rate=DROPOUT_RATE,
        pretrained_embeddings=embedding_matrix,
        freeze_embeddings=False,  # fine-tuning des embeddings
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  → Paramètres entraînables : {n_params:,}")

    # === 5. Entraînement ===
    print("\n--- ÉTAPE 5 : Entraînement ---")
    model, history = train(
        model,
        train_loader,
        val_loader=val_loader,
        lr=LSTM_LEARNING_RATE,
        epochs=LSTM_MAX_EPOCHS,
        weight_decay=0.0,
        use_early_stopping=True,
    )

    # === 6. Évaluation finale sur val ===
    print("\n--- ÉTAPE 6 : Évaluation sur val ---")
    val_results = predict_on_dataloader(model, val_loader)
    val_metrics = compute_metrics(val_results)
    print(f"Métriques val : accuracy={val_metrics['accuracy']:.4f} "
          f"| f1={val_metrics['f1']:.4f}")

    # === 7. Sauvegarde (si production) ===
    if SAVE_AS_PRODUCTION:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), MODELS_DIR / "lstm_classifier.pt")
        with open(MODELS_DIR / "lstm_vocab.pkl", "wb") as f:
            pickle.dump(vocab, f)
        print(f"\nModèle LSTM sauvegardé dans {MODELS_DIR / 'lstm_classifier.pt'}")

    # === 8. Logging ===
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    experiment = {
        "name": EXPERIMENT_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": "imdb",
        "splits": {
            "train": len(train_data),
            "val": len(val_data),
            "test": len(test_data),
        },
        "model": {
            "type": "bilstm",
            "feature_type": "word_sequences",
            "vocab_size": len(vocab),
            "max_seq_len": MAX_SEQ_LEN,
            "embedding_dim": EMBEDDING_DIM,
            "hidden_dim": HIDDEN_DIM,
            "bidirectional": True,
            "dropout_rate": DROPOUT_RATE,
            "pretrained_embeddings": "GloVe 6B.100d",
            "freeze_embeddings": False,
            "n_trainable_params": n_params,
        },
        "preprocessing": {
            "lowercase": True,
            "remove_punctuation": True,
            "min_count": MIN_COUNT,
        },
        "training": {
            "batch_size": LSTM_BATCH_SIZE,
            "max_epochs": LSTM_MAX_EPOCHS,
            "epochs_run": len(history["train_loss"]),
            "learning_rate": LSTM_LEARNING_RATE,
            "optimizer": "Adam",
            "loss": "CrossEntropyLoss",
            "weight_decay": 0.0,
            "device": str(DEVICE),
            "early_stopping": {
                "enabled": True,
                "patience": EARLY_STOPPING_PATIENCE,
                "min_delta": EARLY_STOPPING_MIN_DELTA,
                "stopped_early": history.get("stopped_early", False),
                "best_epoch": history.get("best_epoch", len(history["train_loss"])),
            },
        },
        "val_metrics": val_metrics,
        "loss_history": history,
        "saved_as_production": SAVE_AS_PRODUCTION,
    }
    log_experiment(experiment)

    print("\n" + "=" * 60)
    print("ENTRAÎNEMENT LSTM TERMINÉ")
    print("=" * 60)


if __name__ == "__main__":
    main()