"""
evaluate.py — Évaluation des modèles et calcul des métriques.

Métriques calculées :
- Accuracy : proportion de prédictions correctes
- Precision : parmi les prédictions positives, combien sont vraiment positives
- Recall : parmi les vrais positifs, combien ont été détectés
- F1 : moyenne harmonique de precision et recall
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nlp_sentiment.config import DEVICE


def predict_on_dataloader(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device = DEVICE,
) -> list[dict]:
    """
    Calcule les prédictions du modèle sur tout un DataLoader.

    Args:
        model: Modèle entraîné.
        dataloader: DataLoader contenant les données à évaluer.
        device: 'cpu' ou 'cuda'.

    Returns:
        Liste de dictionnaires avec 'review_id', 'true_label' et 'pred_class'.
    """
    model = model.to(device)
    model.eval()

    results = []
    with torch.no_grad():
        for batch_inputs, batch_labels, batch_ids in dataloader:
            if isinstance(batch_inputs, dict):
                batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
            else:
                batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)

            if isinstance(batch_inputs, dict):
                logits = model(**batch_inputs)
            else:
                logits = model(batch_inputs)

            # .cpu() avant .tolist() : les listes Python vivent sur CPU
            true_labels = batch_labels.argmax(dim=1).cpu().tolist()
            pred_classes = logits.argmax(dim=1).cpu().tolist()

            # batch_ids peut être un tenseur ou un tuple selon le DataLoader
            ids_list = (
                batch_ids.tolist()
                if isinstance(batch_ids, torch.Tensor)
                else list(batch_ids)
            )

            for review_id, true_label, pred_class in zip(
                ids_list, true_labels, pred_classes
            ):
                results.append(
                    {
                        "review_id": review_id,
                        "true_label": true_label,
                        "pred_class": pred_class,
                    }
                )

    return results


def compute_metrics(results: list[dict]) -> dict[str, float]:
    """
    Calcule les métriques de classification binaire.

    Convention : classe 1 = 'pos' (positive), classe 0 = 'neg' (negative).

    Args:
        results: Sortie de `predict_on_dataloader`.

    Returns:
        Dictionnaire avec 'accuracy', 'precision', 'recall', 'f1'.
    """
    true_pos = sum(1 for r in results if r["pred_class"] == 1 and r["true_label"] == 1)
    false_pos = sum(1 for r in results if r["pred_class"] == 1 and r["true_label"] == 0)
    false_neg = sum(1 for r in results if r["pred_class"] == 0 and r["true_label"] == 1)
    true_neg = sum(1 for r in results if r["pred_class"] == 0 and r["true_label"] == 0)

    total = true_pos + false_pos + false_neg + true_neg
    accuracy = (true_pos + true_neg) / total if total > 0 else 0.0

    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0.0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }
    
def compute_loss(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device = DEVICE,
) -> float:
    """
    Calcule la loss moyenne du modèle sur un DataLoader.

    Cette fonction ne modifie pas les poids du modèle (mode eval + no_grad).
    Utilisée pour mesurer la val loss sans déranger l'entraînement.

    Args:
        model: Modèle PyTorch.
        dataloader: DataLoader sur lequel calculer la loss.
        device: 'cpu' ou 'cuda'.

    Returns:
        Loss moyenne sur l'ensemble du DataLoader.
    """
    model = model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch_inputs, batch_labels, _ in dataloader:
            if isinstance(batch_inputs, dict):
                batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
            else:
                batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device).argmax(dim=1)

            if isinstance(batch_inputs, dict):
                logits = model(**batch_inputs)
            else:
                logits = model(batch_inputs)
            loss = criterion(logits, batch_labels)

            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches if n_batches > 0 else 0.0