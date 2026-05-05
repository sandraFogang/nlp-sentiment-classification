"""
train_bert.py — Entraînement et évaluation de DistilBERT.

Fournit `run_bert_experiment()` réutilisable par main() et sweep_bert.py.

Usage Colab (recommandé) :
    !python scripts/train_bert.py

Durée Colab GPU T4 :
- Frozen mode : ~10-15 min (4 epochs)
- Full fine-tuning : ~30-50 min (3 epochs avec early stopping)
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

from nlp_sentiment.bert_data import DEFAULT_BERT_MODEL, prepare_bert_data
from nlp_sentiment.bert_model import BertSentimentClassifier
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
from nlp_sentiment.train import train


# ============================================================================
# Hyperparamètres par défaut spécifiques à BERT
# ============================================================================
# Le LR doit être TRÈS PETIT pour fine-tuner BERT (sinon on casse les poids pré-entraînés)
DEFAULT_BERT_LR_FROZEN = 5e-3   # plus grand car seul le classifier apprend
DEFAULT_BERT_LR_FULL = 2e-5     # standard fine-tuning BERT
DEFAULT_BERT_BATCH_SIZE = 16    # adapté pour T4 GPU avec 512 tokens
DEFAULT_BERT_MAX_EPOCHS = 5     # BERT converge vite


def log_experiment(experiment: dict) -> None:
    """Ajoute une expérience au fichier experiments.json."""
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


def run_bert_experiment(
    experiment_name: str,
    train_data: list[tuple[str, str]],
    val_data: list[tuple[str, str]],
    test_data: list[tuple[str, str]],
    model_name: str = DEFAULT_BERT_MODEL,
    freeze_bert: bool = False,
    max_seq_len: int = 512,
    batch_size: int = DEFAULT_BERT_BATCH_SIZE,
    learning_rate: float | None = None,  # auto selon freeze_bert
    max_epochs: int = DEFAULT_BERT_MAX_EPOCHS,
    dropout_rate: float = 0.3,
    save_as_production_model: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Entraîne et évalue un BertSentimentClassifier.
    """
    # Learning rate adaptatif selon le mode
    if learning_rate is None:
        learning_rate = DEFAULT_BERT_LR_FROZEN if freeze_bert else DEFAULT_BERT_LR_FULL

    if verbose:
        mode = "frozen" if freeze_bert else "full fine-tuning"
        print(f"\n{'#' * 60}")
        print(f"# Expérience BERT : {experiment_name}")
        print(f"# Modèle : {model_name}")
        print(f"# Mode : {mode}, lr={learning_rate}")
        print(f"# Device : {DEVICE}")
        print(f"{'#' * 60}")

    # === 1. Préparation des données ===
    if verbose:
        print("\n--- Préparation des données ---")
    train_dataset, val_dataset, tokenizer = prepare_bert_data(
        train_data, val_data, model_name=model_name, max_seq_len=max_seq_len
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # === 2. Construction du modèle ===
    if verbose:
        print("\n--- Construction du modèle DistilBERT ---")
    model = BertSentimentClassifier(
        model_name=model_name,
        output_dim=len(REVIEW_CLASSES),
        dropout_rate=dropout_rate,
        freeze_bert=freeze_bert,
    )
    n_params = model.count_trainable_params()
    if verbose:
        print(f"  → Paramètres entraînables : {n_params:,}")

    # === 3. Entraînement ===
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

    # === 4. Évaluation sur val ===
    val_results = predict_on_dataloader(model, val_loader)
    val_metrics = compute_metrics(val_results)
    if verbose:
        print(f"\nMétriques val : accuracy={val_metrics['accuracy']:.4f} "
              f"| f1={val_metrics['f1']:.4f}")

    # === 5. Sauvegarde (si production) ===
    if save_as_production_model:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), MODELS_DIR / "bert_classifier.pt")
        # Sauvegarde du tokenizer (par sécurité — il est aussi rechargeable depuis HF)
        tokenizer.save_pretrained(MODELS_DIR / "bert_tokenizer")
        if verbose:
            print(f"Modèle BERT sauvegardé : {MODELS_DIR / 'bert_classifier.pt'}")

    # === 6. Logging ===
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
            "type": f"distilbert_{'frozen' if freeze_bert else 'fine_tuned'}",
            "feature_type": "wordpiece_tokens",
            "model_name": model_name,
            "max_seq_len": max_seq_len,
            "dropout_rate": dropout_rate,
            "freeze_bert": freeze_bert,
            "n_trainable_params": n_params,
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


def main() -> None:
    random.seed(RANDOM_SEED)
    torch.manual_seed(TORCH_SEED)

    print("=" * 60)
    print("ENTRAÎNEMENT BERT (UN SEUL RUN)")
    print("=" * 60)
    train_data, val_data, test_data = load_imdb_splits()
    describe_splits(train_data, val_data, test_data)

    run_bert_experiment(
        experiment_name="distilbert_full_finetune",
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        freeze_bert=False,
        save_as_production_model=False,
    )


if __name__ == "__main__":
    main()