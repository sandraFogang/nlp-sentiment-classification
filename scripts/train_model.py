"""
train_model.py — Script principal d'entraînement.

Lance l'entraînement complet du classificateur bigramme sur le corpus
NLTK Movie Reviews et sauvegarde le modèle dans `models/`.

Usage :
    python scripts/train_model.py
"""
import json
import pickle
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Ajoute la racine du projet au PYTHONPATH pour importer le package src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nlp_sentiment.config import (
    BATCH_SIZE,
    MAX_VOCAB_SIZE,
    MODEL_PATH,
    MODELS_DIR,
    OUTPUTS_DIR,
    RANDOM_SEED,
    REVIEW_CLASSES,
    TORCH_SEED,
    VOCAB_PATH,
)
from nlp_sentiment.data import load_movie_reviews, split_train_test
from nlp_sentiment.evaluate import compute_metrics, predict_on_dataloader
from nlp_sentiment.models import LogisticRegression
from nlp_sentiment.preprocessor import (
    NgramReviewDataset,
    build_ngram_vocab,
    preprocess,
)
from nlp_sentiment.train import train


def main() -> None:
    # === Reproductibilité ===
    random.seed(RANDOM_SEED)
    torch.manual_seed(TORCH_SEED)

    # === 1. Chargement et split du corpus ===
    print("=" * 60)
    print("ÉTAPE 1 — Chargement du corpus NLTK Movie Reviews")
    print("=" * 60)
    documents = load_movie_reviews()
    print(f"Total : {len(documents)} critiques chargées")

    train_dataset, test_dataset = split_train_test(documents)
    print(f"Train : {len(train_dataset)} | Test : {len(test_dataset)}")

    # === 2. Tokenisation ===
    print("\n" + "=" * 60)
    print("ÉTAPE 2 — Tokenisation des critiques")
    print("=" * 60)
    tokenized_train = [(preprocess(text), label) for text, label in train_dataset]
    tokenized_test = [(preprocess(text), label) for text, label in test_dataset]
    print("Tokenisation terminée.")

    # === 3. Construction du vocabulaire bigramme ===
    print("\n" + "=" * 60)
    print("ÉTAPE 3 — Construction du vocabulaire bigramme")
    print("=" * 60)
    bigram_vocab = build_ngram_vocab(tokenized_train, n=2, max_size=MAX_VOCAB_SIZE)
    print(f"Vocabulaire bigramme : {len(bigram_vocab)} bigrammes uniques.")

    # === 4. Création des DataLoaders ===
    train_torch_dataset = NgramReviewDataset(tokenized_train, bigram_vocab, n=2)
    test_torch_dataset = NgramReviewDataset(tokenized_test, bigram_vocab, n=2)
    train_dataloader = DataLoader(train_torch_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_torch_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # === 5. Entraînement ===
    print("\n" + "=" * 60)
    print("ÉTAPE 4 — Entraînement du classificateur bigramme")
    print("=" * 60)
    model = LogisticRegression(
        input_dim=len(bigram_vocab),
        output_dim=len(REVIEW_CLASSES),
    )
    model = train(model, train_dataloader)

    # === 6. Évaluation ===
    print("\n" + "=" * 60)
    print("ÉTAPE 5 — Évaluation sur le test set")
    print("=" * 60)
    results = predict_on_dataloader(model, test_dataloader)
    metrics = compute_metrics(results)
    print("Métriques sur le test set :")
    for key, value in metrics.items():
        print(f"  {key:12s} : {value:.4f}")

    # === 7. Sauvegarde ===
    print("\n" + "=" * 60)
    print("ÉTAPE 6 — Sauvegarde du modèle et des artefacts")
    print("=" * 60)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Modèle PyTorch
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Modèle sauvegardé : {MODEL_PATH}")

    # Vocabulaire (pickle)
    with open(VOCAB_PATH, "wb") as f:
        pickle.dump(bigram_vocab, f)
    print(f"Vocabulaire sauvegardé : {VOCAB_PATH}")

    # Métriques (JSON pour le README)
    metrics_path = OUTPUTS_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "Logistic Regression (bigrammes)",
                "vocab_size": len(bigram_vocab),
                "train_size": len(train_dataset),
                "test_size": len(test_dataset),
                "metrics": metrics,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Métriques sauvegardées : {metrics_path}")

    print("\nEntraînement terminé avec succès.")


if __name__ == "__main__":
    main()