"""
sweep_l2.py — Grid search sur la régularisation L2 (weight_decay).

Ce script :
1. Charge IMDB une seule fois (gain de temps)
2. Lance 4 entraînements avec différentes valeurs de weight_decay
3. Logge chaque run dans experiments.json
4. Affiche un tableau comparatif final

Usage :
    python scripts/sweep_l2.py

Durée : ~25-30 minutes au total (5-7 min par run × 4 runs).
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

# Import depuis train_model.py (pas un module mais on traite scripts/ comme un)
from train_model import run_experiment


# ============================================================================
# Configuration du sweep
# ============================================================================
# Liste des valeurs de weight_decay à tester
WEIGHT_DECAYS_TO_TEST = [0.0, 1e-5, 1e-4, 1e-3]


# ============================================================================
# Pipeline du sweep
# ============================================================================
def main() -> None:
    # Reproductibilité globale
    random.seed(RANDOM_SEED)
    torch.manual_seed(TORCH_SEED)

    # Chargement IMDB (une seule fois pour les 4 runs)
    print("=" * 60)
    print("CHARGEMENT IMDB (une seule fois pour tout le sweep)")
    print("=" * 60)
    train_data, val_data, test_data = load_imdb_splits()
    describe_splits(train_data, val_data, test_data)

    # Stockage des résultats pour le tableau final
    results = []

    print("\n" + "=" * 60)
    print(f"SWEEP : {len(WEIGHT_DECAYS_TO_TEST)} runs à enchaîner")
    print(f"Valeurs à tester : {WEIGHT_DECAYS_TO_TEST}")
    print("=" * 60)

    for i, wd in enumerate(WEIGHT_DECAYS_TO_TEST, start=1):
        # Re-fixer les graines à chaque run pour comparaison équitable
        random.seed(RANDOM_SEED)
        torch.manual_seed(TORCH_SEED)

        # Nom standardisé : bigram_l2_0e0, bigram_l2_1e-5, etc.
        if wd == 0.0:
            name = "bigram_l2_baseline_0"
        else:
            name = f"bigram_l2_{wd:.0e}"

        print(f"\n[{i}/{len(WEIGHT_DECAYS_TO_TEST)}] Lancement : {name}")

        experiment = run_experiment(
            experiment_name=name,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            ngram_n=2,
            weight_decay=wd,
            save_as_production_model=False,
            verbose=True,
        )
        results.append(experiment)

    # Tableau comparatif final
    print("\n" + "=" * 60)
    print("TABLEAU COMPARATIF DU SWEEP L2")
    print("=" * 60)
    print(
        f"{'Expérience':<25} {'weight_decay':<14} "
        f"{'Val acc':<10} {'Val F1':<10} {'Min val loss':<14} {'Best epoch':<12}"
    )
    print("-" * 90)
    for exp in results:
        wd = exp["training"]["weight_decay"]
        acc = exp["val_metrics"]["accuracy"]
        f1 = exp["val_metrics"]["f1"]
        val_losses = exp["loss_history"]["val_loss"]
        min_val_loss = min(val_losses)
        best_epoch = val_losses.index(min_val_loss) + 1
        print(
            f"{exp['name']:<25} {wd:<14.0e} "
            f"{acc:<10.4f} {f1:<10.4f} {min_val_loss:<14.4f} {best_epoch:<12}"
        )

    # Identification du meilleur run
    best = max(results, key=lambda e: e["val_metrics"]["accuracy"])
    print(
        f"\n→ Meilleur modèle : {best['name']} "
        f"(accuracy val = {best['val_metrics']['accuracy']:.4f})"
    )


if __name__ == "__main__":
    main()