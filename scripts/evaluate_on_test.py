"""evaluate_on_test.py — Final evaluation of champion models on the IMDB test set.

This is the ONE-SHOT evaluation : the test set has never been seen during
development. The numbers produced here are the unbiased estimates to report
in the README and CV.

Usage :
    python scripts/evaluate_on_test.py              # both models
    python scripts/evaluate_on_test.py --tfidf      # TF-IDF only (fast, CPU)
    python scripts/evaluate_on_test.py --bert       # BERT only (GPU recommended)

Duration :
    TF-IDF : ~30 seconds (CPU)
    BERT   : ~5-8 minutes (GPU T4) or ~30-45 minutes (CPU)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ============================================================================
# Helpers
# ============================================================================
def print_metrics(name: str, y_true, y_pred, n_examples: int) -> dict:
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_pos = f1_score(y_true, y_pred, pos_label=1, average="binary")
    f1_neg = f1_score(y_true, y_pred, pos_label=0, average="binary")
    cm = confusion_matrix(y_true, y_pred)

    print("\n" + "=" * 60)
    print(f"  TEST SET EVALUATION : {name}")
    print("=" * 60)
    print(f"  Examples       : {n_examples:,}")
    print(f"  Accuracy       : {acc:.4f}  ({acc * 100:.2f}%)")
    print(f"  F1 macro       : {f1_macro:.4f}  ({f1_macro * 100:.2f}%)")
    print(f"  F1 (positive)  : {f1_pos:.4f}")
    print(f"  F1 (negative)  : {f1_neg:.4f}")
    print()
    print("  Confusion matrix :")
    print(f"                  Predicted")
    print(f"                  Neg     Pos")
    print(f"  Actual Neg    {cm[0][0]:>6}  {cm[0][1]:>6}")
    print(f"  Actual Pos    {cm[1][0]:>6}  {cm[1][1]:>6}")
    print()
    print("  Classification report :")
    print(classification_report(
        y_true, y_pred,
        target_names=["Negative", "Positive"],
        digits=4,
    ))

    return {
        "model": name,
        "n_examples": n_examples,
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_positive": float(f1_pos),
        "f1_negative": float(f1_neg),
        "confusion_matrix": cm.tolist(),
    }


# ============================================================================
# TF-IDF evaluation
# ============================================================================
def evaluate_tfidf(test_data) -> dict:
    """Evaluate the TF-IDF champion on the full test set."""
    from nlp_sentiment.config import MODEL_PATH, VOCAB_PATH
    from nlp_sentiment.predict import predict

    if not (MODEL_PATH.exists() and VOCAB_PATH.exists()):
        print(f"❌ TF-IDF model not found at {MODEL_PATH}")
        print("   Run : python scripts/train_champion_tfidf.py")
        return None

    print("\n→ Evaluating TF-IDF on the full test set (25,000 reviews)...")
    print("  This takes about 30 seconds on CPU.\n")

    start = time.time()
    y_true = []
    y_pred = []

    for i, (text, label) in enumerate(test_data, start=1):
        result = predict(text)
        y_true.append(1 if label == "pos" else 0)
        y_pred.append(1 if result["label"] == "positif" else 0)
        if i % 2500 == 0:
            elapsed = time.time() - start
            rate = i / elapsed
            eta = (len(test_data) - i) / rate
            print(f"  Processed {i:,} / {len(test_data):,} "
                  f"({rate:.0f} reviews/s, ETA {eta:.0f}s)")

    elapsed = time.time() - start
    print(f"\n  Total time : {elapsed:.1f}s "
          f"({len(test_data) / elapsed:.0f} reviews/s)")

    return print_metrics("TF-IDF + Logistic Regression",
                         y_true, y_pred, len(test_data))


# ============================================================================
# BERT evaluation
# ============================================================================
def evaluate_bert(test_data) -> dict:
    """Evaluate the DistilBERT champion on the full test set."""
    from torch.utils.data import DataLoader

    from nlp_sentiment.bert_data import BertReviewDataset
    from nlp_sentiment.bert_model import BertSentimentClassifier
    from transformers import AutoTokenizer

    BERT_MODEL_PATH = PROJECT_ROOT / "models" / "checkpoints" / "distilbert_full_finetune.pt"
    BERT_TOKENIZER_PATH = PROJECT_ROOT / "models" / "checkpoints" / "distilbert_tokenizer"

    if not BERT_MODEL_PATH.exists():
        print(f"❌ BERT model not found at {BERT_MODEL_PATH}")
        print("   Train and download from Colab via train_champion_bert.py")
        return None

    if not BERT_TOKENIZER_PATH.exists():
        print(f"❌ BERT tokenizer not found at {BERT_TOKENIZER_PATH}")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n→ Evaluating DistilBERT on the full test set (25,000 reviews)...")
    print(f"  Device : {device}")
    if device.type == "cpu":
        print("  ⚠️  CPU will take ~30-45 minutes. GPU strongly recommended.")
    print()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(str(BERT_TOKENIZER_PATH))
    model = BertSentimentClassifier(
        model_name="distilbert-base-uncased",
        freeze_bert=False,
        dropout_rate=0.3,
    )
    model.load_state_dict(torch.load(BERT_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Build the test dataset
    texts = [t for t, _ in test_data]
    labels = [1 if l == "pos" else 0 for _, l in test_data]
    test_dataset = BertReviewDataset(
        texts=texts, labels=labels, tokenizer=tokenizer, max_seq_len=512,
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Run inference
    start = time.time()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].cpu().numpy()

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

            y_true.extend(batch_labels.tolist())
            y_pred.extend(preds.tolist())

            if i % 100 == 0:
                processed = i * 32
                elapsed = time.time() - start
                rate = processed / elapsed
                eta = (len(test_data) - processed) / rate
                print(f"  Processed {processed:,} / {len(test_data):,} "
                      f"({rate:.0f} reviews/s, ETA {eta:.0f}s)")

    elapsed = time.time() - start
    print(f"\n  Total time : {elapsed:.1f}s "
          f"({len(test_data) / elapsed:.0f} reviews/s)")

    return print_metrics("DistilBERT (full fine-tuning)",
                         y_true, y_pred, len(test_data))


# ============================================================================
# Main
# ============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tfidf", action="store_true", help="Evaluate only TF-IDF")
    parser.add_argument("--bert", action="store_true", help="Evaluate only BERT")
    args = parser.parse_args()

    # Default : both
    eval_tfidf = args.tfidf or not (args.tfidf or args.bert)
    eval_bert = args.bert or not (args.tfidf or args.bert)

    print("=" * 60)
    print("FINAL EVALUATION ON IMDB TEST SET (25,000 REVIEWS)")
    print("=" * 60)
    print("This is a one-shot evaluation. The test set has never been seen")
    print("during development. The numbers below are the unbiased final scores.")
    print()

    # Load the test set
    from nlp_sentiment.data import load_imdb_splits
    print("Loading data...")
    _, _, test_data = load_imdb_splits()
    print(f"Test set : {len(test_data):,} reviews")

    results = {}

    if eval_tfidf:
        result = evaluate_tfidf(test_data)
        if result is not None:
            results["tfidf"] = result

    if eval_bert:
        result = evaluate_bert(test_data)
        if result is not None:
            results["bert"] = result

    # Save results
    output_path = PROJECT_ROOT / "outputs" / "test_set_results.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if "tfidf" in results:
        print(f"  TF-IDF        : Acc {results['tfidf']['accuracy'] * 100:.2f}%  "
              f"F1 {results['tfidf']['f1_macro'] * 100:.2f}%")
    if "bert" in results:
        print(f"  DistilBERT    : Acc {results['bert']['accuracy'] * 100:.2f}%  "
              f"F1 {results['bert']['f1_macro'] * 100:.2f}%")
    print()
    print(f"Results saved to : {output_path}")


if __name__ == "__main__":
    main()