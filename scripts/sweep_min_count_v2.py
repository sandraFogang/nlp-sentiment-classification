"""
sweep_min_count_v2.py — Re-run K=3 et K=5 avec early stopping et checkpointing.

But : valider que l'infrastructure d'early stopping fonctionne et vérifier
si entraîner plus longtemps (avec restauration du meilleur modèle) améliore
les résultats par rapport au sweep initial (5 époques fixes).

Usage :
    python scripts/sweep_min_count_v2.py

Durée : ~15-25 minutes (2 runs avec early stopping).
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


# Re-tester les 2 meilleurs candidats du sweep initial
MIN_COUNT_VALUES = [3]


def main() -> None:
    random.seed(RANDOM_SEED)
    torch.manual_seed(TORCH_SEED)

    print("=" * 60)
    print("RE-RUN AVEC EARLY STOPPING (K=3, K=5)")
    print("=" * 60)
    train_data, val_data, test_data = load_imdb_splits()
    describe_splits(train_data, val_data, test_data)

    results = []

    for i, k in enumerate(MIN_COUNT_VALUES, start=1):
        random.seed(RANDOM_SEED)
        torch.manual_seed(TORCH_SEED)

        name = f"bigram_mincount_{k}_es"  # _es = early stopping

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
            max_epochs=15,
            use_early_stopping=True,
            save_as_production_model=True,
            verbose=True,
        )
        results.append(experiment)

    # Tableau comparatif
    print("\n" + "=" * 60)
    print("RÉSULTATS — RE-RUN AVEC EARLY STOPPING")
    print("=" * 60)
    print(
        f"{'Expérience':<28} {'Vocab':<10} {'Best epoch':<12} "
        f"{'Stopped':<10} {'Val acc':<10} {'Val F1':<10} {'Val loss':<10}"
    )
    print("-" * 95)

    # Référence : ancien runs sans early stopping
    print("Anciens runs (5 époques fixes, pas d'early stopping) :")
    print(
        f"{'  bigram_mincount_3':<28} {'214756':<10} {'5/5':<12} {'--':<10} "
        f"{'0.9067':<10} {'0.9064':<10} {'0.2353':<10}"
    )
    print(
        f"{'  bigram_mincount_5':<28} {'121350':<10} {'5/5':<12} {'--':<10} "
        f"{'0.9007':<10} {'0.9008':<10} {'0.2407':<10}"
    )
    print()
    print("Nouveaux runs (early stopping + checkpointing) :")
    for exp in results:
        vsize = exp["model"]["vocab_size"]
        epochs_run = exp["training"]["epochs_run"]
        max_eps = exp["training"]["max_epochs"]
        best_epoch = exp["training"]["early_stopping"]["best_epoch"]
        stopped = "Oui" if exp["training"]["early_stopping"]["stopped_early"] else "Non"
        acc = exp["val_metrics"]["accuracy"]
        f1 = exp["val_metrics"]["f1"]
        val_losses = exp["loss_history"]["val_loss"]
        min_val = min(val_losses)
        print(
            f"  {exp['name']:<26} {vsize:<10} {f'{best_epoch}/{epochs_run}':<12} "
            f"{stopped:<10} {acc:<10.4f} {f1:<10.4f} {min_val:<10.4f}"
        )

    best = max(results, key=lambda e: e["val_metrics"]["f1"])
    print(
        f"\n→ Meilleur run (sélection par F1) : {best['name']} "
        f"(F1 val = {best['val_metrics']['f1']:.4f})"
    )


if __name__ == "__main__":
    main()