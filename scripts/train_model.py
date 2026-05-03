"""
train_model.py — Script d'entraînement et d'évaluation sur le val set.

Ce script :
1. Charge IMDB et prépare train/val
2. Entraîne le modèle spécifié (par défaut : bigramme baseline)
3. Évalue sur le VAL set uniquement (jamais sur le test)
4. Sauvegarde le modèle de production (s'il s'agit du modèle déployable)
5. Logge l'expérience dans outputs/experiments.json

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

# Permet d'importer le package `nlp_sentiment` même sans installation editable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nlp_sentiment.config import (
    BATCH_SIZE,
    EXPERIMENTS_PATH,
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
    build_ngram_vocab,
    preprocess,
)
from nlp_sentiment.train import train


# ============================================================================
# Configuration de l'expérience
# ============================================================================
# Nom unique pour identifier cette expérience dans l'historique
EXPERIMENT_NAME = "bigram_baseline_imdb"

# Si True, ce modèle sera sauvegardé dans models/ pour le déploiement
# (À mettre à False pour des expériences de test que tu ne veux pas déployer)
SAVE_AS_PRODUCTION_MODEL = True

# Hyperparamètres de l'expérience (modifiables pour les variantes)
NGRAM_N = 2  # 1 = unigrammes, 2 = bigrammes, 3 = trigrammes


# ============================================================================
# Fonctions utilitaires
# ============================================================================
def log_experiment(experiment: dict) -> None:
    """
    Ajoute une expérience au fichier experiments.json.

    Crée le fichier s'il n'existe pas. Ajoute à la suite sinon.
    """
    EXPERIMENTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    if EXPERIMENTS_PATH.exists():
        with open(EXPERIMENTS_PATH, "r", encoding="utf-8") as f:
            history = json.load(f)
    else:
        history = []

    history.append(experiment)

    with open(EXPERIMENTS_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"\nExpérience enregistrée dans : {EXPERIMENTS_PATH}")


# ============================================================================
# Pipeline principal
# ============================================================================
def main() -> None:
    # --- Reproductibilité ---
    random.seed(RANDOM_SEED)
    torch.manual_seed(TORCH_SEED)

    # --- 1. Chargement IMDB et splits ---
    print("=" * 60)
    print("ÉTAPE 1 — Chargement IMDB (train + val + test)")
    print("=" * 60)
    train_data, val_data, test_data = load_imdb_splits()
    describe_splits(train_data, val_data, test_data)
    print(f"\nNote : le test set ({len(test_data)} critiques) ne sera PAS utilisé ici.")
    print("Il est réservé à l'évaluation finale du meilleur modèle.")

    # --- 2. Tokenisation ---
    print("\n" + "=" * 60)
    print("ÉTAPE 2 — Tokenisation")
    print("=" * 60)
    print("Tokenisation du train...")
    tokenized_train = [(preprocess(text), label) for text, label in train_data]
    print("Tokenisation du val...")
    tokenized_val = [(preprocess(text), label) for text, label in val_data]
    print("Tokenisation terminée.")

    # --- 3. Construction du vocabulaire (uniquement à partir du train) ---
    print("\n" + "=" * 60)
    print(f"ÉTAPE 3 — Vocabulaire {NGRAM_N}-grammes (max {MAX_VOCAB_SIZE} tokens)")
    print("=" * 60)
    ngram_vocab = build_ngram_vocab(
        tokenized_train, n=NGRAM_N, max_size=MAX_VOCAB_SIZE
    )
    print(f"Vocabulaire construit : {len(ngram_vocab)} {NGRAM_N}-grammes uniques.")

    # --- 4. Création des DataLoaders ---
    train_torch_ds = NgramReviewDataset(tokenized_train, ngram_vocab, n=NGRAM_N)
    val_torch_ds = NgramReviewDataset(tokenized_val, ngram_vocab, n=NGRAM_N)
    train_loader = DataLoader(train_torch_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_torch_ds, batch_size=BATCH_SIZE, shuffle=False)

    # --- 5. Entraînement ---
    print("\n" + "=" * 60)
    print(f"ÉTAPE 4 — Entraînement (régression logistique sur {NGRAM_N}-grammes)")
    print("=" * 60)
    model = LogisticRegression(
        input_dim=len(ngram_vocab),
        output_dim=len(REVIEW_CLASSES),
    )
    model = train(model, train_loader)

    # --- 6. Évaluation sur le VAL set (pas le test) ---
    print("\n" + "=" * 60)
    print("ÉTAPE 5 — Évaluation sur le VAL set")
    print("=" * 60)
    val_results = predict_on_dataloader(model, val_loader)
    val_metrics = compute_metrics(val_results)
    print("Métriques sur le val set :")
    for key, value in val_metrics.items():
        print(f"  {key:12s} : {value:.4f}")

    # --- 7. Sauvegarde du modèle (si modèle de production) ---
    if SAVE_AS_PRODUCTION_MODEL:
        print("\n" + "=" * 60)
        print("ÉTAPE 6 — Sauvegarde du modèle de production")
        print("=" * 60)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        with open(VOCAB_PATH, "wb") as f:
            pickle.dump(ngram_vocab, f)
        print(f"Modèle      : {MODEL_PATH}")
        print(f"Vocabulaire : {VOCAB_PATH}")
    else:
        print("\nModèle non sauvegardé (SAVE_AS_PRODUCTION_MODEL = False).")

    # --- 8. Logging de l'expérience ---
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
            "type": "logistic_regression",
            "features": f"{NGRAM_N}-grams (count)",
            "vocab_size": len(ngram_vocab),
            "max_vocab_size": MAX_VOCAB_SIZE,
        },
        "preprocessing": {
            "lowercase": True,
            "remove_punctuation": True,
            "stop_words_removed": False,
            "lemmatization": False,
        },
        "training": {
            "batch_size": BATCH_SIZE,
            "epochs": 5,
            "learning_rate": 0.001,
            "optimizer": "Adam",
            "loss": "CrossEntropyLoss",
        },
        "val_metrics": val_metrics,
        "saved_as_production": SAVE_AS_PRODUCTION_MODEL,
    }
    log_experiment(experiment)

    print("\n" + "=" * 60)
    print("PIPELINE TERMINÉ")
    print("=" * 60)


if __name__ == "__main__":
    main()