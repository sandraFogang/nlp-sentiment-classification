"""sweep_bert.py — Compare frozen vs full fine-tuning + auto-save the winner."""
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
        "description": "DistilBERT frozen — only the classifier is trained",
        "freeze_bert": True,
    },
    {
        "name": "distilbert_full_finetune",
        "description": "DistilBERT full fine-tuning — entire model trained",
        "freeze_bert": False,
    },
]


def main() -> None:
    random.seed(RANDOM_SEED)
    torch.manual_seed(TORCH_SEED)

    print("=" * 60)
    print("BERT SWEEP : Frozen vs Full Fine-tuning (auto-save winner)")
    print("=" * 60)

    train_data, val_data, test_data = load_imdb_splits()
    describe_splits(train_data, val_data, test_data)

    # === Pass 1 : Run both experiments without saving ===
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
        results.append((config, experiment))

    # === Comparative table ===
    print("\n\n" + "=" * 60)
    print("BERT COMPARISON TABLE")
    print("=" * 60)
    print(
        f"{'Experiment':<32} {'Mode':<18} {'Params':<12} "
        f"{'Best ep.':<10} {'Val acc':<10} {'Val F1':<10}"
    )
    print("-" * 95)
    print("For reference :")
    print(
        f"{'  tfidf_uni_bi_sublinear':<32} {'TF-IDF':<18} {'~0.5M':<12} "
        f"{'23/26':<10} {'0.9200':<10} {'0.9192':<10}"
    )
    print(
        f"{'  bilstm_v2_full_glove300':<32} {'LSTM + GloVe 300d':<18} {'10.9M':<12} "
        f"{'2/5':<10} {'0.9137':<10} {'0.9140':<10}"
    )
    print()
    print("New runs (DistilBERT) :")
    for config, exp in results:
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

    # === Pass 2 : Re-train and save the winner ===
    best_config, best_exp = max(results, key=lambda r: r[1]["val_metrics"]["f1"])

    print(f"\n\n→ Winner : {best_exp['name']} (F1 val = {best_exp['val_metrics']['f1']:.4f})")
    print("=" * 60)
    print(f"RE-TRAINING THE WINNER ({best_exp['name']}) WITH SAVE")
    print("=" * 60)

    random.seed(RANDOM_SEED)
    torch.manual_seed(TORCH_SEED)

    final_exp = run_bert_experiment(
        experiment_name=f"{best_config['name']}_PROD",
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        freeze_bert=best_config["freeze_bert"],
        save_as_production_model=True,
        verbose=True,
    )

    print("\n\n" + "=" * 60)
    print("BERT CHAMPION SAVED")
    print("=" * 60)
    print(f"  Mode         : {'Frozen' if best_config['freeze_bert'] else 'Full fine-tune'}")
    print(f"  Val accuracy : {final_exp['val_metrics']['accuracy']:.4f}")
    print(f"  Val F1       : {final_exp['val_metrics']['f1']:.4f}")
    print()
    print("Files generated (Colab) :")
    print("  - models/bert_classifier.pt")
    print("  - models/bert_tokenizer/")
    print()
    print("To archive locally after download :")
    print('  Copy-Item "models\\bert_classifier.pt" "models\\checkpoints\\distilbert_winner.pt"')
    print("  # Then download the entire bert_tokenizer/ folder via Colab files.download")


if __name__ == "__main__":
    main()