"""
train_model.py — Script d'entraînement et d'évaluation sur le val set.

Ce script entraîne UN modèle avec des hyperparamètres fixes et
optionnellement le sauvegarde comme modèle de production.

Pour comparer plusieurs hyperparamètres, voir scripts/sweep_*.py.

Usage :
    python scripts/train_model.py
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
    EARLY_STOPPING_MIN_DELTA,
    EARLY_STOPPING_PATIENCE,
    EXPERIMENTS_PATH,
    LEARNING_RATE,
    MAX_EPOCHS,
    MAX_VOCAB_SIZE,
    MODEL_PATH,
    MODELS_DIR,
    OUTPUTS_DIR,
    RANDOM_SEED,
    REVIEW_CLASSES,
    TORCH_SEED,
    VOCAB_PATH,
)
from nlp_sentiment.data import describe_splits, load_imdb_splits
from nlp_sentiment.evaluate import compute_metrics, predict_on_dataloader
from nlp_sentiment.models import LogisticRegression
from nlp_sentiment.preprocessor import (
    NgramReviewDataset,
    TfidfReviewDataset,
    build_ngram_vocab,
    build_ngram_vocab_min_count,
    preprocess,
)
from nlp_sentiment.train import train


# ============================================================================
# Fonctions utilitaires
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


def run_experiment(
    experiment_name: str,
    train_data: list[tuple[str, str]],
    val_data: list[tuple[str, str]],
    test_data: list[tuple[str, str]],
    ngram_n: int = 2,
    vocab_strategy: str = "top_k",
    vocab_param: int = MAX_VOCAB_SIZE,
    feature_type: str = "count",
    ngram_range: tuple[int, int] = (2, 2),
    sublinear_tf: bool = False,
    weight_decay: float = 0.0,
    max_epochs: int = MAX_EPOCHS,
    use_early_stopping: bool = True,
    save_as_production_model: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Entraîne et évalue un modèle (n-grammes count ou TF-IDF).

    Args:
        experiment_name: Identifiant unique de l'expérience.
        train_data, val_data, test_data: Splits IMDB.
        ngram_n: 1=unigrammes, 2=bigrammes (mode count uniquement).
        vocab_strategy: 'top_k' ou 'min_count' (mode count uniquement).
        vocab_param: max_size (top_k), min_count (min_count) ou min_df (tfidf).
        feature_type: 'count' ou 'tfidf'.
        ngram_range: ex (1,2) pour uni+bi (mode tfidf uniquement).
        sublinear_tf: applique log(1+TF) (mode tfidf uniquement).
        weight_decay: Régularisation L2.
        max_epochs: Nb max d'époques (early stopping peut arrêter plus tôt).
        use_early_stopping: Active l'early stopping.
        save_as_production_model: Sauvegarde dans models/ comme prod.
        verbose: Affiche les étapes.

    Returns:
        Le dict de l'expérience (logué dans experiments.json).
    """
    if verbose:
        print(f"\n{'#' * 60}")
        print(f"# Expérience : {experiment_name}")
        print(f"# weight_decay = {weight_decay}")
        print(f"{'#' * 60}")

    # === Tokenisation ===
    if verbose:
        print("\nTokenisation du train et du val...")
    tokenized_train = [(preprocess(text), label) for text, label in train_data]
    tokenized_val = [(preprocess(text), label) for text, label in val_data]

    # === Construction des features (deux modes) ===
    ngram_vocab = None  # utilisé uniquement en mode count, sera None en tfidf

    if feature_type == "count":
        if verbose:
            print(f"Construction du vocabulaire {ngram_n}-grammes "
                  f"(stratégie={vocab_strategy}, paramètre={vocab_param})...")

        if vocab_strategy == "top_k":
            ngram_vocab = build_ngram_vocab(
                tokenized_train, n=ngram_n, max_size=vocab_param
            )
        elif vocab_strategy == "min_count":
            ngram_vocab = build_ngram_vocab_min_count(
                tokenized_train, n=ngram_n, min_count=vocab_param
            )
        else:
            raise ValueError(
                f"vocab_strategy inconnue : {vocab_strategy}. "
                f"Utilisez 'top_k' ou 'min_count'."
            )

        if verbose:
            print(f"  → {len(ngram_vocab)} {ngram_n}-grammes uniques.")

        train_torch_ds = NgramReviewDataset(tokenized_train, ngram_vocab, n=ngram_n)
        val_torch_ds = NgramReviewDataset(tokenized_val, ngram_vocab, n=ngram_n)
        vocab_size = len(ngram_vocab)

    elif feature_type == "tfidf":
        if verbose:
            print(f"Construction des features TF-IDF "
                  f"(ngram_range={ngram_range}, min_df={vocab_param}, "
                  f"sublinear_tf={sublinear_tf})...")

        train_torch_ds = TfidfReviewDataset(
            tokenized_train,
            vectorizer=None,
            ngram_range=ngram_range,
            min_df=vocab_param,
            sublinear_tf=sublinear_tf,
        )
        val_torch_ds = TfidfReviewDataset(
            tokenized_val,
            vectorizer=train_torch_ds.vectorizer,
            ngram_range=ngram_range,
            min_df=vocab_param,
            sublinear_tf=sublinear_tf,
        )
        vocab_size = train_torch_ds.vocab_size

        if verbose:
            print(f"  → {vocab_size} features TF-IDF.")
    else:
        raise ValueError(
            f"feature_type inconnu : {feature_type}. "
            f"Utilisez 'count' ou 'tfidf'."
        )

    # === DataLoaders ===
    train_loader = DataLoader(train_torch_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_torch_ds, batch_size=BATCH_SIZE, shuffle=False)

    # === Modèle ===
    model = LogisticRegression(
        input_dim=vocab_size,
        output_dim=len(REVIEW_CLASSES),
    )

    # === Entraînement ===
    if verbose:
        print(f"\nEntraînement (weight_decay={weight_decay})...")
    model, history = train(
        model,
        train_loader,
        val_loader=val_loader,
        epochs=max_epochs,
        weight_decay=weight_decay,
        use_early_stopping=use_early_stopping,
    )

    # === Évaluation sur val ===
    val_results = predict_on_dataloader(model, val_loader)
    val_metrics = compute_metrics(val_results)

    if verbose:
        print(f"\nMétriques val : accuracy={val_metrics['accuracy']:.4f} "
              f"| f1={val_metrics['f1']:.4f}")

    # === Sauvegarde du modèle (production) ===
    if save_as_production_model:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)

        if feature_type == "count":
            with open(VOCAB_PATH, "wb") as f:
                pickle.dump(ngram_vocab, f)
        elif feature_type == "tfidf":
            with open(VOCAB_PATH, "wb") as f:
                pickle.dump(train_torch_ds.vectorizer, f)

        if verbose:
            print(f"Modèle sauvegardé comme production : {MODEL_PATH}")

    # === Logging de l'expérience ===
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
            "type": "logistic_regression",
            "feature_type": feature_type,
            "features": (
                f"{ngram_range[0]}-{ngram_range[1]}-grams ({feature_type})"
                if feature_type == "tfidf"
                else f"{ngram_n}-grams (count)"
            ),
            "vocab_size": vocab_size,
            "vocab_strategy": vocab_strategy,
            "vocab_param": vocab_param,
            "sublinear_tf": sublinear_tf if feature_type == "tfidf" else None,
        },
        "preprocessing": {
            "lowercase": True,
            "remove_punctuation": True,
            "stop_words_removed": False,
            "lemmatization": False,
        },
        "training": {
            "batch_size": BATCH_SIZE,
            "max_epochs": max_epochs,
            "epochs_run": len(history["train_loss"]),
            "learning_rate": LEARNING_RATE,
            "optimizer": "Adam",
            "loss": "CrossEntropyLoss",
            "weight_decay": weight_decay,
            "early_stopping": {
                "enabled": use_early_stopping,
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
# Pipeline principal (un seul run)
# ============================================================================
def main() -> None:
    random.seed(RANDOM_SEED)
    torch.manual_seed(TORCH_SEED)

    print("=" * 60)
    print("Chargement IMDB (train + val + test)")
    print("=" * 60)
    train_data, val_data, test_data = load_imdb_splits()
    describe_splits(train_data, val_data, test_data)

    run_experiment(
        experiment_name="bigram_baseline_imdb",
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        ngram_n=2,
        vocab_strategy="min_count",
        vocab_param=3,
        feature_type="count",
        weight_decay=0.0,
        save_as_production_model=False,
    )


if __name__ == "__main__":
    main()