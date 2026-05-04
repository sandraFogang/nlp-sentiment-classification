"""
train_champion_tfidf.py — Entraîne le modèle champion TF-IDF avec sauvegarde.

Configuration : TF-IDF sur unigrammes + bigrammes combinés, sublinear_tf=True.
Ce modèle a obtenu val accuracy=0.9200 / F1=0.9192 dans le sweep TF-IDF.

Usage :
    python scripts/train_champion_tfidf.py

Durée : ~15 minutes (early stopping vers epoch 23).
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

from train_model import run_experiment


def main() -> None:
    random.seed(RANDOM_SEED)
    torch.manual_seed(TORCH_SEED)

    print("=" * 60)
    print("ENTRAÎNEMENT DU CHAMPION TF-IDF")
    print("=" * 60)
    train_data, val_data, test_data = load_imdb_splits()
    describe_splits(train_data, val_data, test_data)

    experiment = run_experiment(
        experiment_name="tfidf_uni_bi_sublinear_PROD",
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        feature_type="tfidf",
        ngram_range=(1, 2),
        vocab_param=3,
        sublinear_tf=True,
        weight_decay=0.0,
        max_epochs=45,
        use_early_stopping=True,
        save_as_production_model=True,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("MODÈLE SAUVEGARDÉ COMME PRODUCTION")
    print("=" * 60)
    print(f"  Val accuracy : {experiment['val_metrics']['accuracy']:.4f}")
    print(f"  Val F1       : {experiment['val_metrics']['f1']:.4f}")
    print(f"  Best epoch   : {experiment['training']['early_stopping']['best_epoch']}")
    print()
    print("Pensez à archiver le modèle dans models/checkpoints/ :")
    print('  Copy-Item "models\\classifier.pt" "models\\checkpoints\\tfidf_uni_bi_sublinear.pt"')
    print('  Copy-Item "models\\vocab.pkl" "models\\checkpoints\\tfidf_uni_bi_sublinear_vectorizer.pkl"')


if __name__ == "__main__":
    main()