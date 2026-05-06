"""train_champion_bert.py — Train and save the BERT champion.

Configuration : DistilBERT full fine-tuning (winner from sweep_bert.py).
Best result : val accuracy=0.9303, F1=0.9319.

Usage on Colab :
    !python scripts/train_champion_bert.py

Duration : ~25-30 minutes on T4 GPU.
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

from train_bert import run_bert_experiment


def main() -> None:
    random.seed(RANDOM_SEED)
    torch.manual_seed(TORCH_SEED)

    print("=" * 60)
    print("TRAINING THE BERT CHAMPION (full fine-tuning)")
    print("=" * 60)

    train_data, val_data, test_data = load_imdb_splits()
    describe_splits(train_data, val_data, test_data)

    experiment = run_bert_experiment(
        experiment_name="distilbert_full_finetune_PROD",
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        model_name="distilbert-base-uncased",
        freeze_bert=False,
        max_seq_len=512,
        batch_size=16,
        max_epochs=5,
        dropout_rate=0.3,
        save_as_production_model=True,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("BERT CHAMPION SAVED")
    print("=" * 60)
    print(f"  Val accuracy : {experiment['val_metrics']['accuracy']:.4f}")
    print(f"  Val F1       : {experiment['val_metrics']['f1']:.4f}")
    print(f"  Best epoch   : {experiment['training']['early_stopping']['best_epoch']}")
    print()
    print("Files generated (Colab) :")
    print("  - models/bert_classifier.pt")
    print("  - models/bert_tokenizer/")


if __name__ == "__main__":
    main()