"""
sweep_lstm_vocab.py — Grid search sur la taille du vocabulaire LSTM.

Compare 3 tailles : 15k, 30k, 60k mots.
Tous les autres hyperparamètres sont fixes.
GloVe est chargé UNE SEULE FOIS et réutilisé (gain de temps).

Usage local (CPU, très lent — déconseillé) :
    python scripts/sweep_lstm_vocab.py

Usage Colab (GPU, recommandé) :
    Voir notebook Colab fourni séparément.

Durée estimée :
    - Sur CPU : 4-8 heures (à éviter)
    - Sur Colab GPU T4 : 30-50 minutes
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

from train_lstm import DEFAULT_EMBEDDING_DIM, DEFAULT_GLOVE_PATH, run_lstm_experiment


# ============================================================================
# Configuration du sweep
# ============================================================================
VOCAB_SIZES_TO_TEST = [15000, 30000, 60000]


def main() -> None:
    random.seed(RANDOM_SEED)
    torch.manual_seed(TORCH_SEED)

    print("=" * 60)
    print("SWEEP LSTM : TAILLE DU VOCABULAIRE")
    print(f"Valeurs à tester : {VOCAB_SIZES_TO_TEST}")
    print("=" * 60)

    # === Chargement IMDB une seule fois ===
    print("\n--- Chargement IMDB ---")
    train_data, val_data, test_data = load_imdb_splits()
    describe_splits(train_data, val_data, test_data)

    # === Chargement de GloVe une seule fois (réutilisé pour les 3 runs) ===
    print("\n--- Chargement de GloVe (une seule fois pour les 3 runs) ---")
    glove_embeddings = load_glove_embeddings(DEFAULT_GLOVE_PATH, DEFAULT_EMBEDDING_DIM)

    results = []

    for i, vocab_size in enumerate(VOCAB_SIZES_TO_TEST, start=1):
        # Re-fixer les graines à chaque run pour comparaison équitable
        random.seed(RANDOM_SEED)
        torch.manual_seed(TORCH_SEED)

        name = f"bilstm_glove_vocab_{vocab_size // 1000}k"

        print(f"\n\n[{i}/{len(VOCAB_SIZES_TO_TEST)}] Lancement : {name}")
        print(f"     max_vocab_size = {vocab_size}")

        experiment = run_lstm_experiment(
            experiment_name=name,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            max_vocab_size=vocab_size,
            glove_embeddings=glove_embeddings,  # réutilisation
            save_as_production_model=False,
            verbose=True,
        )
        results.append(experiment)

    # === Tableau comparatif final ===
    print("\n\n" + "=" * 60)
    print("TABLEAU COMPARATIF DU SWEEP LSTM (taille du vocabulaire)")
    print("=" * 60)
    print(
        f"{'Expérience':<28} {'max_vocab':<12} {'vocab actuel':<14} "
        f"{'best epoch':<12} {'val acc':<10} {'val F1':<10}"
    )
    print("-" * 90)
    for exp in results:
        max_v = exp["model"]["max_vocab_size"]
        actual_v = exp["model"]["vocab_size"]
        best_ep = exp["training"]["early_stopping"]["best_epoch"]
        epochs_run = exp["training"]["epochs_run"]
        acc = exp["val_metrics"]["accuracy"]
        f1 = exp["val_metrics"]["f1"]
        print(
            f"{exp['name']:<28} {max_v:<12,} {actual_v:<14,} "
            f"{f'{best_ep}/{epochs_run}':<12} {acc:<10.4f} {f1:<10.4f}"
        )

    # Identifier le meilleur run par F1
    best = max(results, key=lambda e: e["val_metrics"]["f1"])
    print(
        f"\n→ Meilleur run (par F1) : {best['name']} "
        f"(vocab actuel = {best['model']['vocab_size']:,}, "
        f"F1 val = {best['val_metrics']['f1']:.4f})"
    )

    # Comparaison avec le champion TF-IDF (rappel)
    print(f"\n→ Pour mémoire, champion TF-IDF : F1 val = 0.9192 (uni+bi sublinear)")


if __name__ == "__main__":
    main()