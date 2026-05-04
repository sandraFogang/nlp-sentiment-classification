"""
sweep_tfidf.py — Comparaison de variantes TF-IDF.

Tests :
1. TF-IDF sur bigrammes seuls (équivalent direct à K=3 count)
2. TF-IDF sur uni+bi combinés (covers two birds avec une pierre)
3. TF-IDF avec sublinear_tf (variante standard sklearn)

Usage :
    python scripts/sweep_tfidf.py

Durée : ~20-25 minutes au total (avec early stopping).
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


# ============================================================================
# Configuration du sweep
# ============================================================================
EXPERIMENTS = [
    {
        "name": "tfidf_bi_min3",
        "ngram_range": (2, 2),
        "sublinear_tf": False,
        "description": "TF-IDF sur bigrammes seuls (référence)",
    },
    {
        "name": "tfidf_uni_bi_min3",
        "ngram_range": (1, 2),
        "sublinear_tf": False,
        "description": "TF-IDF sur unigrammes + bigrammes combinés",
    },
    {
        "name": "tfidf_uni_bi_sublinear",
        "ngram_range": (1, 2),
        "sublinear_tf": True,
        "description": "TF-IDF uni+bi avec sublinear_tf=True",
    },
]


def main() -> None:
    random.seed(RANDOM_SEED)
    torch.manual_seed(TORCH_SEED)

    print("=" * 60)
    print("SWEEP TF-IDF")
    print("=" * 60)
    train_data, val_data, test_data = load_imdb_splits()
    describe_splits(train_data, val_data, test_data)

    results = []

    print("\n" + "=" * 60)
    print(f"{len(EXPERIMENTS)} configurations à tester")
    print("=" * 60)

    for i, config in enumerate(EXPERIMENTS, start=1):
        random.seed(RANDOM_SEED)
        torch.manual_seed(TORCH_SEED)

        print(f"\n[{i}/{len(EXPERIMENTS)}] {config['name']}")
        print(f"     → {config['description']}")

        experiment = run_experiment(
            experiment_name=config["name"],
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            feature_type="tfidf",
            ngram_range=config["ngram_range"],
            vocab_param=3,  # min_df=3 pour cohérence avec les sweeps précédents
            sublinear_tf=config["sublinear_tf"],
            weight_decay=0.0,
            max_epochs=45,
            use_early_stopping=True,
            save_as_production_model=False,
            verbose=True,
        )
        results.append((config, experiment))

    # Tableau comparatif
    print("\n" + "=" * 60)
    print("RÉSULTATS — SWEEP TF-IDF")
    print("=" * 60)
    print(
        f"{'Expérience':<28} {'ngram_range':<14} {'sublinear':<11} "
        f"{'Vocab':<10} {'Best ep.':<10} {'Val acc':<10} {'Val F1':<10}"
    )
    print("-" * 105)

    # Référence : meilleur run en mode comptage
    print("Référence (mode comptage) :")
    print(
        f"{'  bigram_mincount_3_es':<28} {'(2,2)':<14} {'-':<11} "
        f"{'214756':<10} {'5/8':<10} {'0.9067':<10} {'0.9064':<10}"
    )
    print()
    print("Nouveaux runs (mode TF-IDF) :")
    for config, exp in results:
        ngrange = f"{config['ngram_range']}"
        sublinear = "Oui" if config["sublinear_tf"] else "Non"
        vsize = exp["model"]["vocab_size"]
        epochs_run = exp["training"]["epochs_run"]
        best_epoch = exp["training"]["early_stopping"]["best_epoch"]
        acc = exp["val_metrics"]["accuracy"]
        f1 = exp["val_metrics"]["f1"]
        print(
            f"  {exp['name']:<26} {ngrange:<14} {sublinear:<11} "
            f"{vsize:<10} {f'{best_epoch}/{epochs_run}':<10} "
            f"{acc:<10.4f} {f1:<10.4f}"
        )

    best = max(results, key=lambda r: r[1]["val_metrics"]["f1"])
    print(
        f"\n→ Meilleur run TF-IDF (sélection par F1) : {best[1]['name']} "
        f"(F1 val = {best[1]['val_metrics']['f1']:.4f})"
    )


if __name__ == "__main__":
    main()