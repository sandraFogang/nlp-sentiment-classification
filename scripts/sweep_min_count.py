"""
sweep_min_count.py — Grid search sur le seuil de fréquence min_count.

Compare la stratégie de vocabulaire 'top_k' (baseline) avec différentes
valeurs du seuil 'min_count' (3, 5, 10, 20).

Usage :
    python scripts/sweep_min_count.py

Durée : ~30-40 minutes au total. K=3 est plus lent (vocab plus gros).
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
# Liste des valeurs de min_count à tester
# (le baseline top-K=30000 est déjà dans experiments.json, on ne le refait pas)
MIN_COUNT_VALUES = [3, 5, 10, 20]


def main() -> None:
    random.seed(RANDOM_SEED)
    torch.manual_seed(TORCH_SEED)

    # Chargement IMDB une seule fois
    print("=" * 60)
    print("CHARGEMENT IMDB (une seule fois pour tout le sweep)")
    print("=" * 60)
    train_data, val_data, test_data = load_imdb_splits()
    describe_splits(train_data, val_data, test_data)

    results = []

    print("\n" + "=" * 60)
    print(f"SWEEP MIN_COUNT : {len(MIN_COUNT_VALUES)} runs")
    print(f"Valeurs à tester : {MIN_COUNT_VALUES}")
    print("=" * 60)

    for i, k in enumerate(MIN_COUNT_VALUES, start=1):
        random.seed(RANDOM_SEED)
        torch.manual_seed(TORCH_SEED)

        name = f"bigram_mincount_{k}"

        print(f"\n[{i}/{len(MIN_COUNT_VALUES)}] Lancement : {name}")

        experiment = run_experiment(
            experiment_name=name,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            ngram_n=2,
            vocab_strategy="min_count",
            vocab_param=k,
            weight_decay=0.0,
            save_as_production_model=False,
            verbose=True,
        )
        results.append(experiment)

    # Tableau comparatif
    print("\n" + "=" * 60)
    print("TABLEAU COMPARATIF — STRATÉGIE MIN_COUNT")
    print("=" * 60)
    print(
        f"{'Expérience':<25} {'min_count':<10} {'Vocab size':<12} "
        f"{'Val acc':<10} {'Val F1':<10} {'Min val loss':<14} {'Best epoch':<12}"
    )
    print("-" * 95)

    # Ajout du baseline pour comparaison visuelle (depuis experiments.json)
    print(
        f"{'bigram_l2_baseline_0':<25} {'(top-30k)':<10} "
        f"{'30000':<12} {'0.8853':<10} {'0.8847':<10} {'0.2662':<14} {'3':<12}  ← référence"
    )

    for exp in results:
        k = exp["model"]["vocab_param"]
        vsize = exp["model"]["vocab_size"]
        acc = exp["val_metrics"]["accuracy"]
        f1 = exp["val_metrics"]["f1"]
        val_losses = exp["loss_history"]["val_loss"]
        min_val_loss = min(val_losses)
        best_epoch = val_losses.index(min_val_loss) + 1
        print(
            f"{exp['name']:<25} {k:<10} {vsize:<12} "
            f"{acc:<10.4f} {f1:<10.4f} {min_val_loss:<14.4f} {best_epoch:<12}"
        )

    # Sélection par F1 (plus robuste que l'accuracy en cas de déséquilibre)
    best = max(results, key=lambda e: e["val_metrics"]["f1"])
    print(
        f"\n→ Meilleur run min_count (sélection par F1) : {best['name']} "
        f"(min_count={best['model']['vocab_param']}, "
        f"F1 val = {best['val_metrics']['f1']:.4f}, "
        f"accuracy val = {best['val_metrics']['accuracy']:.4f})"
    )


if __name__ == "__main__":
    main()