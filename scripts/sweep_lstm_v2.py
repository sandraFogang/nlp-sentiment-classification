"""
sweep_lstm_v2.py — Ablation complète LSTM (5 runs).

Compare 5 configurations en isolant l'impact de chaque amélioration :
1. Baseline (déjà fait dans le sweep précédent — 89.37%)
2. Hidden 256 (1 couche, last pooling, GloVe 100d)
3. Hidden 256 + 2 couches
4. Hidden 256 + 2 couches + mean pooling
5. Hidden 256 + 2 couches + mean pooling + GloVe 300d (le combo final)

Usage Colab (recommandé) :
    !python scripts/sweep_lstm_v2.py

Durée estimée sur Colab T4 : 25-40 minutes (4 nouveaux runs).
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


# ============================================================================
# Configuration des 4 nouveaux runs (le baseline est déjà fait)
# ============================================================================
EXPERIMENTS = [
    {
        "name": "bilstm_v2_hidden256",
        "description": "Levier 2a : hidden=256, 1 couche, last pooling, GloVe 100d",
        "model_version": "v2",
        "hidden_dim": 256,
        "num_layers": 1,
        "pooling": "last",
        "embedding_dim": 100,
        "glove_filename": "glove.6B.100d.txt",
    },
    {
        "name": "bilstm_v2_hidden256_2layers",
        "description": "Levier 2b : hidden=256, 2 couches, last pooling, GloVe 100d",
        "model_version": "v2",
        "hidden_dim": 256,
        "num_layers": 2,
        "pooling": "last",
        "embedding_dim": 100,
        "glove_filename": "glove.6B.100d.txt",
    },
    {
        "name": "bilstm_v2_hidden256_2layers_meanpool",
        "description": "Levier 2c : hidden=256, 2 couches, mean pooling, GloVe 100d",
        "model_version": "v2",
        "hidden_dim": 256,
        "num_layers": 2,
        "pooling": "mean",
        "embedding_dim": 100,
        "glove_filename": "glove.6B.100d.txt",
    },
    {
        "name": "bilstm_v2_full_glove300",
        "description": "Combo final : hidden=256, 2 couches, mean pooling, GloVe 300d",
        "model_version": "v2",
        "hidden_dim": 256,
        "num_layers": 2,
        "pooling": "mean",
        "embedding_dim": 300,
        "glove_filename": "glove.6B.300d.txt",
    },
]

# Hyperparamètres communs à tous les runs
COMMON_PARAMS = {
    "max_vocab_size": 30000,
    "min_count": 5,
    "max_seq_len": 512,
    "dropout_rate": 0.3,
    "lstm_dropout": 0.3,
    "learning_rate": 0.001,
    "batch_size": 64,
    "max_epochs": 15,
}


def main() -> None:
    random.seed(RANDOM_SEED)
    torch.manual_seed(TORCH_SEED)

    print("=" * 60)
    print("SWEEP LSTM V2 : ABLATION COMPLÈTE (4 nouveaux runs)")
    print("=" * 60)

    train_data, val_data, test_data = load_imdb_splits()
    describe_splits(train_data, val_data, test_data)

    # Pré-chargement de GloVe : 100d ET 300d (mais 300d uniquement si nécessaire)
    print("\n--- Chargement de GloVe 100d (pour les runs 1-3) ---")
    glove_path_100 = PROJECT_ROOT / "data" / "glove" / "glove.6B.100d.txt"
    glove_100 = load_glove_embeddings(glove_path_100, 100)

    glove_300 = None  # chargé seulement quand on en a besoin

    results = []

    for i, config in enumerate(EXPERIMENTS, start=1):
        random.seed(RANDOM_SEED)
        torch.manual_seed(TORCH_SEED)

        print(f"\n\n[{i}/{len(EXPERIMENTS)}] {config['name']}")
        print(f"     → {config['description']}")

        # Choix de GloVe selon la dimension
        if config["embedding_dim"] == 100:
            glove_to_use = glove_100
            glove_path = glove_path_100
        else:  # 300d
            if glove_300 is None:
                print("\n--- Chargement de GloVe 300d (premier run qui en a besoin) ---")
                glove_path_300 = PROJECT_ROOT / "data" / "glove" / "glove.6B.300d.txt"
                glove_300 = load_glove_embeddings(glove_path_300, 300)
            glove_to_use = glove_300
            glove_path = PROJECT_ROOT / "data" / "glove" / config["glove_filename"]

        experiment = run_lstm_experiment(
            experiment_name=config["name"],
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            embedding_dim=config["embedding_dim"],
            hidden_dim=config["hidden_dim"],
            model_version=config["model_version"],
            num_layers=config["num_layers"],
            pooling=config["pooling"],
            glove_embeddings=glove_to_use,
            glove_path=glove_path,
            save_as_production_model=False,
            verbose=True,
            **COMMON_PARAMS,
        )
        results.append((config, experiment))

    # === Tableau d'ablation final ===
    print("\n\n" + "=" * 60)
    print("TABLEAU D'ABLATION LSTM")
    print("=" * 60)
    print(
        f"{'Expérience':<42} {'Hidden':<8} {'Lay':<5} {'Pool':<6} "
        f"{'GloVe':<7} {'Params':<11} {'Val acc':<10} {'Val F1':<10}"
    )
    print("-" * 110)

    # Référence : baseline du sweep précédent
    print("Référence (sweep précédent) :")
    print(
        f"{'  bilstm_glove_vocab_15k (baseline)':<42} {'64':<8} {'1':<5} {'last':<6} "
        f"{'100d':<7} {'1.6M':<11} {'0.8937':<10} {'0.8934':<10}"
    )
    print()
    print("Ablation V2 :")
    for config, exp in results:
        h = exp["model"]["hidden_dim"]
        nl = exp["model"]["num_layers"]
        pool = exp["model"]["pooling"]
        emb = exp["model"]["embedding_dim"]
        params = exp["model"]["n_trainable_params"]
        params_str = f"{params / 1e6:.1f}M"
        acc = exp["val_metrics"]["accuracy"]
        f1 = exp["val_metrics"]["f1"]
        print(
            f"  {exp['name']:<40} {h:<8} {nl:<5} {pool:<6} "
            f"{emb}d{'':<3} {params_str:<11} {acc:<10.4f} {f1:<10.4f}"
        )

    # Meilleur run par F1
    best = max(results, key=lambda r: r[1]["val_metrics"]["f1"])
    print(
        f"\n→ Meilleur run V2 : {best[1]['name']} "
        f"(F1 val = {best[1]['val_metrics']['f1']:.4f})"
    )
    print(f"→ Pour mémoire, champion TF-IDF : F1 val = 0.9192")


if __name__ == "__main__":
    main()