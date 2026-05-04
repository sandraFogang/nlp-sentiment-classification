"""
train_lstm.py — Entraînement du LSTM bidirectionnel avec GloVe.

Ce module fournit une fonction `run_lstm_experiment()` réutilisable par :
- main() : un seul run avec les paramètres par défaut
- scripts/sweep_lstm_vocab.py : sweep sur la taille du vocabulaire
- (futur) scripts/sweep_lstm_hidden.py, etc.

Usage local (CPU, lent) :
    python scripts/train_lstm.py

Usage Colab (GPU, rapide) :
    Voir notebook Colab fourni séparément.
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
    DEVICE,
    EARLY_STOPPING_MIN_DELTA,
    EARLY_STOPPING_PATIENCE,
    EXPERIMENTS_PATH,
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
# Constantes par défaut (peuvent être surchargées par les paramètres de fonction)
# ============================================================================
DEFAULT_GLOVE_PATH = PROJECT_ROOT / "data" / "glove" / "glove.6B.100d.txt"
DEFAULT_EMBEDDING_DIM = 100
DEFAULT_MAX_SEQ_LEN = 512
DEFAULT_MIN_COUNT = 5
DEFAULT_HIDDEN_DIM = 64
DEFAULT_DROPOUT_RATE = 0.3
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_EPOCHS = 15


# ============================================================================
# Logging des expériences
# ============================================================================
def log_experiment(experiment: dict) -> None:
    """Ajoute une expérience au fichier experiments.json (cumulatif)."""
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
# Pipeline réutilisable : un run d'entraînement LSTM
# ============================================================================
def run_lstm_experiment(
    experiment_name: str,
    train_data: list[tuple[str, str]],
    val_data: list[tuple[str, str]],
    test_data: list[tuple[str, str]],
    max_vocab_size: int = 30000,
    min_count: int = DEFAULT_MIN_COUNT,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    dropout_rate: float = DEFAULT_DROPOUT_RATE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    glove_embeddings: dict[str, torch.Tensor] | None = None,
    glove_path: Path = DEFAULT_GLOVE_PATH,
    save_as_production_model: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Entraîne et évalue un BiLSTMClassifier avec embeddings GloVe.

    Args:
        experiment_name: Identifiant unique de l'expérience.
        train_data, val_data, test_data: Splits IMDB.
        max_vocab_size: Taille max du vocabulaire LSTM (incluant <PAD> et <UNK>).
        min_count: Fréquence minimale d'un mot.
        max_seq_len: Longueur des séquences (padding/troncature).
        embedding_dim: Dimension des embeddings (doit matcher GloVe).
        hidden_dim: Dimension de l'état caché du LSTM (par direction).
        dropout_rate: Taux de dropout.
        learning_rate, batch_size, max_epochs: Hyperparamètres d'entraînement.
        glove_embeddings: Dict GloVe pré-chargé (optimisation pour les sweeps).
                          Si None, recharge depuis glove_path.
        glove_path: Chemin du fichier GloVe (utilisé si glove_embeddings=None).
        save_as_production_model: Si True, sauvegarde dans models/.
        verbose: Affiche les étapes.

    Returns:
        Le dict de l'expérience (logué dans experiments.json).
    """
    if verbose:
        print(f"\n{'#' * 60}")
        print(f"# Expérience LSTM : {experiment_name}")
        print(f"# vocab_size_max = {max_vocab_size}, hidden_dim = {hidden_dim}")
        print(f"# Device : {DEVICE}")
        print(f"{'#' * 60}")

    # === 1. Préparation des séquences ===
    if verbose:
        print("\n--- Préparation des séquences ---")
    train_dataset, val_dataset, vocab = prepare_lstm_data(
        train_data,
        val_data,
        max_vocab_size=max_vocab_size,
        min_count=min_count,
        max_seq_len=max_seq_len,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # === 2. Chargement de GloVe (si pas déjà fourni) ===
    if glove_embeddings is None:
        if verbose:
            print("\n--- Chargement de GloVe ---")
        glove_embeddings = load_glove_embeddings(glove_path, embedding_dim)
    elif verbose:
        print("\n--- GloVe déjà chargé (réutilisation) ---")

    embedding_matrix = build_embedding_matrix(vocab, glove_embeddings, embedding_dim)

    # === 3. Construction du modèle ===
    if verbose:
        print("\n--- Construction du BiLSTM ---")
    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        emb_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=len(REVIEW_CLASSES),
        dropout_rate=dropout_rate,
        pretrained_embeddings=embedding_matrix,
        freeze_embeddings=False,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"  → Paramètres entraînables : {n_params:,}")

    # === 4. Entraînement ===
    if verbose:
        print("\n--- Entraînement ---")
    model, history = train(
        model,
        train_loader,
        val_loader=val_loader,
        lr=learning_rate,
        epochs=max_epochs,
        weight_decay=0.0,
        use_early_stopping=True,
    )

    # === 5. Évaluation sur val ===
    val_results = predict_on_dataloader(model, val_loader)
    val_metrics = compute_metrics(val_results)
    if verbose:
        print(f"\nMétriques val : accuracy={val_metrics['accuracy']:.4f} "
              f"| f1={val_metrics['f1']:.4f}")

    # === 6. Sauvegarde (si production) ===
    if save_as_production_model:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), MODELS_DIR / "lstm_classifier.pt")
        with open(MODELS_DIR / "lstm_vocab.pkl", "wb") as f:
            pickle.dump(vocab, f)
        if verbose:
            print(f"Modèle LSTM sauvegardé : {MODELS_DIR / 'lstm_classifier.pt'}")

    # === 7. Logging ===
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    experiment = {
        "name": experiment_name,
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
            "max_vocab_size": max_vocab_size,
            "max_seq_len": max_seq_len,
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "bidirectional": True,
            "dropout_rate": dropout_rate,
            "pretrained_embeddings": "GloVe 6B.100d",
            "freeze_embeddings": False,
            "n_trainable_params": n_params,
        },
        "preprocessing": {
            "lowercase": True,
            "remove_punctuation": True,
            "min_count": min_count,
        },
        "training": {
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "epochs_run": len(history["train_loss"]),
            "learning_rate": learning_rate,
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
        "saved_as_production": save_as_production_model,
    }
    log_experiment(experiment)

    return experiment


# ============================================================================
# Pipeline principal (un seul run avec les paramètres par défaut)
# ============================================================================
def main() -> None:
    random.seed(RANDOM_SEED)
    torch.manual_seed(TORCH_SEED)

    print("=" * 60)
    print("ENTRAÎNEMENT LSTM (UN SEUL RUN)")
    print("=" * 60)
    train_data, val_data, test_data = load_imdb_splits()
    describe_splits(train_data, val_data, test_data)

    run_lstm_experiment(
        experiment_name="bilstm_glove_30k",
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        max_vocab_size=30000,
        save_as_production_model=False,
    )

    print("\n" + "=" * 60)
    print("ENTRAÎNEMENT TERMINÉ")
    print("=" * 60)


if __name__ == "__main__":
    main()