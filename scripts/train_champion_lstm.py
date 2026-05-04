"""
train_champion_lstm.py — Entraîne et sauvegarde le LSTM champion.

Configuration : hidden=256, 2 couches, mean pooling, GloVe 300d.
Ce modèle a obtenu val F1=0.9140 dans l'ablation V2 (sweep_lstm_v2.py).

Usage Colab (recommandé) :
    !python scripts/train_champion_lstm.py

Durée : ~5-7 minutes sur GPU T4.

À la fin :
- models/lstm_classifier.pt        (poids du modèle)
- models/lstm_vocab.pkl            (vocabulaire)
- outputs/experiments.json         (entrée mise à jour)
"""
import random
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from nlp_sentiment.config import RANDOM_SEED, TORCH_SEED
from nlp_sentiment.data import describe_splits, load_imdb_splits
from nlp_sentiment.glove_utils import load_glove_embeddings

from train_lstm import run_lstm_experiment


def main() -> None:
    random.seed(RANDOM_SEED)
    torch.manual_seed(TORCH_SEED)

    print("=" * 60)
    print("ENTRAÎNEMENT DU CHAMPION LSTM")
    print("Configuration : hidden=256, 2 couches, mean pooling, GloVe 300d")
    print("=" * 60)

    # === Chargement IMDB ===
    train_data, val_data, test_data = load_imdb_splits()
    describe_splits(train_data, val_data, test_data)

    # === Chargement de GloVe 300d ===
    print("\n--- Chargement de GloVe 300d ---")
    glove_path = PROJECT_ROOT / "data" / "glove" / "glove.6B.300d.txt"
    glove_embeddings = load_glove_embeddings(glove_path, 300)

    # === Entraînement et sauvegarde ===
    experiment = run_lstm_experiment(
        experiment_name="bilstm_v2_full_glove300_PROD",
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        max_vocab_size=30000,
        min_count=5,
        max_seq_len=512,
        embedding_dim=300,
        hidden_dim=256,
        dropout_rate=0.3,
        learning_rate=0.001,
        batch_size=64,
        max_epochs=15,
        glove_embeddings=glove_embeddings,
        glove_path=glove_path,
        model_version="v2",
        num_layers=2,
        pooling="mean",
        lstm_dropout=0.3,
        save_as_production_model=True,  # ← clé : on sauvegarde !
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("CHAMPION LSTM SAUVEGARDÉ")
    print("=" * 60)
    print(f"  Val accuracy : {experiment['val_metrics']['accuracy']:.4f}")
    print(f"  Val F1       : {experiment['val_metrics']['f1']:.4f}")
    print(f"  Best epoch   : {experiment['training']['early_stopping']['best_epoch']}")
    print()
    print("Fichiers générés (Colab) :")
    print("  - models/lstm_classifier.pt")
    print("  - models/lstm_vocab.pkl")
    print()
    print("Pour archiver localement (après téléchargement) :")
    print('  Copy-Item "models\\lstm_classifier.pt" "models\\checkpoints\\bilstm_v2_full_glove300.pt"')
    print('  Copy-Item "models\\lstm_vocab.pkl" "models\\checkpoints\\bilstm_v2_full_glove300_vocab.pkl"')


if __name__ == "__main__":
    main()