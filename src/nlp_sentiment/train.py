"""
train.py — Boucle d'entraînement avec early stopping et checkpointing.

La fonction `train` :
- Entraîne le modèle sur `dataloader` (train set)
- Évalue la val loss à chaque époque
- Sauvegarde le meilleur modèle (par val loss) en mémoire
- Arrête l'entraînement si la val loss ne baisse plus pendant `patience` époques
- Restaure les poids du meilleur modèle à la fin
"""
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from nlp_sentiment.config import (
    DEVICE,
    EARLY_STOPPING_MIN_DELTA,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE,
    MAX_EPOCHS,
)
from nlp_sentiment.evaluate import compute_loss


def train(
    model: nn.Module,
    dataloader: DataLoader,
    val_loader: DataLoader | None = None,
    lr: float = LEARNING_RATE,
    epochs: int = MAX_EPOCHS,
    device: torch.device = DEVICE,
    weight_decay: float = 0.0,
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE,
    early_stopping_min_delta: float = EARLY_STOPPING_MIN_DELTA,
    use_early_stopping: bool = True,
) -> tuple[nn.Module, dict]:
    """
    Entraîne un modèle PyTorch avec early stopping et checkpointing.

    Si val_loader est fourni et use_early_stopping=True, l'entraînement :
    - Calcule la val loss à chaque époque
    - Garde en mémoire les poids du modèle ayant la meilleure val loss
    - Arrête si la val loss ne baisse plus pendant `early_stopping_patience` époques
    - Retourne le modèle avec les poids du MEILLEUR état (pas le dernier)

    Sans val_loader ou avec use_early_stopping=False, comportement classique :
    entraîne pour `epochs` époques et retourne le modèle final.

    Args:
        model: Le modèle à entraîner.
        dataloader: DataLoader train.
        val_loader: DataLoader val (requis pour early stopping).
        lr: Taux d'apprentissage.
        epochs: Nombre maximum d'époques (sera atteint sans early stopping).
        device: 'cpu' ou 'cuda'.
        weight_decay: Coefficient de régularisation L2.
        early_stopping_patience: Nb d'époques sans amélioration avant arrêt.
        early_stopping_min_delta: Amélioration minimale pour réinitialiser la patience.
        use_early_stopping: Si False, désactive l'arrêt précoce (boucle complète).

    Returns:
        Tuple (model, history) où :
        - model : meilleur modèle trouvé (poids restaurés si early stopping actif)
        - history : dict avec 'train_loss', 'val_loss', 'best_epoch', 'stopped_early'
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    history = {
        "train_loss": [],
        "val_loss": [],
        "best_epoch": 0,
        "stopped_early": False,
    }

    # === État de l'early stopping ===
    best_val_loss = float("inf")
    best_model_state = None
    epochs_without_improvement = 0

    can_early_stop = (val_loader is not None) and use_early_stopping

    for epoch in range(epochs):
        # === Phase d'entraînement ===
        model.train()
        train_loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        epoch_loss = 0.0

        for batch_inputs, batch_labels, _ in train_loop:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device).argmax(dim=1)

            optimizer.zero_grad()
            logits = model(batch_inputs)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            train_loop.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = epoch_loss / len(dataloader)
        history["train_loss"].append(round(avg_train_loss, 4))

        # === Phase de validation ===
        if val_loader is not None:
            avg_val_loss = compute_loss(model, val_loader, device)
            history["val_loss"].append(round(avg_val_loss, 4))

            # === Vérification de l'amélioration ===
            improvement = best_val_loss - avg_val_loss
            is_improved = improvement > early_stopping_min_delta

            if is_improved:
                # Sauvegarde du meilleur état (deep copy pour éviter de partager les tenseurs)
                best_val_loss = avg_val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                history["best_epoch"] = epoch + 1
                epochs_without_improvement = 0
                marker = " ← meilleur"
            else:
                epochs_without_improvement += 1
                marker = f" (patience {epochs_without_improvement}/{early_stopping_patience})"

            print(
                f"Epoch {epoch + 1}/{epochs} — "
                f"train loss : {avg_train_loss:.4f} | "
                f"val loss : {avg_val_loss:.4f}{marker}"
            )

            # === Test d'arrêt précoce ===
            if can_early_stop and epochs_without_improvement >= early_stopping_patience:
                print(
                    f"\n→ Early stopping déclenché à l'epoch {epoch + 1} "
                    f"(pas d'amélioration depuis {early_stopping_patience} époques)."
                )
                print(f"→ Meilleur modèle : epoch {history['best_epoch']} "
                      f"(val loss = {best_val_loss:.4f})")
                history["stopped_early"] = True
                break
        else:
            print(f"Epoch {epoch + 1}/{epochs} — train loss : {avg_train_loss:.4f}")

    # === Restauration du meilleur modèle ===
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if not history["stopped_early"]:
            print(f"\n→ Entraînement terminé. Meilleur modèle : epoch {history['best_epoch']}.")

    return model, history