"""
sweep_bert.py — Comparaison frozen vs full fine-tuning DistilBERT.

Tests :
1. Frozen BERT (classifier seul entraîné)
2. Full fine-tuning (tout BERT + classifier)

Usage Colab :
    !python scripts/sweep_bert.py

Durée Colab GPU T4 : ~45-60 minutes au total.
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


EXPERIMENTS = [
    {
        "name": "distilbert_frozen",
        "description": "DistilBERT frozen — seul le classifier est entraîné",
        "freeze_bert": True,
    },
    {
        "name": "distilbert_full_finetune",
        "description": "DistilBERT full fine-tuning — tout le modèle est entraîné",
        "freeze_bert": False,
    },
]


def main() -> None:
    random.seed(RANDOM_SEED)
    torch.manual_seed(TORCH_SEED)

    print("=" * 60)
    print("SWEEP BERT : Frozen vs Full Fine-tuning")
    print("=" * 60)
    train_data, val_data, test_data = load_imdb_splits()
    describe_splits(train_data, val_data, test_data)

    results = []

    for i, config in enumerate(EXPERIMENTS, start=1):
        random.seed(RANDOM_SEED)
        torch.manual_seed(TORCH_SEED)

        print(f"\n\n[{i}/{len(EXPERIMENTS)}] {config['name']}")
        print(f"     → {config['description']}")

        experiment = run_bert_experiment(
            experiment_name=config["name"],
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            freeze_bert=config["freeze_bert"],
            save_as_production_model=False,
            verbose=True,
        )
        results.append(experiment)

    # Tableau comparatif
    print("\n\n" + "=" * 60)
    print("TABLEAU COMPARATIF — DISTILBERT FROZEN vs FULL")
    print("=" * 60)
    print(
        f"{'Expérience':<32} {'Mode':<18} {'Params':<12} "
        f"{'Best ep.':<10} {'Val acc':<10} {'Val F1':<10}"
    )
    print("-" * 95)

    # Référence : champion TF-IDF + champion LSTM
    print("Pour mémoire :")
    print(
        f"{'  tfidf_uni_bi_sublinear':<32} {'TF-IDF':<18} {'~0.5M':<12} "
        f"{'23/26':<10} {'0.9200':<10} {'0.9192':<10}"
    )
    print(
        f"{'  bilstm_v2_full_glove300':<32} {'LSTM + GloVe 300d':<18} {'10.9M':<12} "
        f"{'2/5':<10} {'0.9137':<10} {'0.9140':<10}"
    )
    print()
    print("Nouveaux runs (DistilBERT) :")
    for exp in results:
        mode = "Frozen" if exp["model"]["freeze_bert"] else "Full fine-tune"
        params = f"{exp['model']['n_trainable_params'] / 1e6:.1f}M"
        epochs_run = exp["training"]["epochs_run"]
        best_ep = exp["training"]["early_stopping"]["best_epoch"]
        acc = exp["val_metrics"]["accuracy"]
        f1 = exp["val_metrics"]["f1"]
        print(
            f"  {exp['name']:<30} {mode:<18} {params:<12} "
            f"{f'{best_ep}/{epochs_run}':<10} {acc:<10.4f} {f1:<10.4f}"
        )

    best = max(results, key=lambda e: e["val_metrics"]["f1"])
    print(f"\n→ Meilleur run BERT : {best['name']} "
          f"(F1 val = {best['val_metrics']['f1']:.4f})")


if __name__ == "__main__":
    main()