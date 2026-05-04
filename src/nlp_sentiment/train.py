"""
train.py — Boucle d'entraînement avec suivi de la val loss.

La fonction `train` :
- Entraîne le modèle sur `dataloader` (train set)
- Optionnellement, calcule la val loss à chaque époque sur `val_loader`
- Retourne le modèle entraîné + l'historique des losses train/val
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from nlp_sentiment.config import DEVICE, LEARNING_RATE, N_EPOCHS
from nlp_sentiment.evaluate import compute_loss


def train(
    model: nn.Module,
    dataloader: DataLoader,
    val_loader: DataLoader | None = None,
    lr: float = LEARNING_RATE,
    epochs: int = N_EPOCHS,
    device: torch.device = DEVICE,
    weight_decay: float = 0.0,
) -> tuple[nn.Module, dict]:
    """
    Entraîne un modèle PyTorch et retourne l'historique des losses.

    Args:
        model: Le modèle à entraîner.
        dataloader: DataLoader contenant les données d'entraînement.
        val_loader: DataLoader de validation (optionnel). Si fourni, calcule
                    la val loss à chaque époque.
        lr: Taux d'apprentissage.
        epochs: Nombre de passes sur les données.
        device: 'cpu' ou 'cuda'.
        weight_decay: Coefficient de régularisation L2 (0 = pas de régularisation).

    Returns:
        Tuple (model, history) où history est un dict avec :
        - 'train_loss' : liste des losses train par époque
        - 'val_loss' : liste des losses val par époque (vide si val_loader=None)
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": []}

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

        # === Phase de validation (si val_loader fourni) ===
        if val_loader is not None:
            avg_val_loss = compute_loss(model, val_loader, device)
            history["val_loss"].append(round(avg_val_loss, 4))
            print(
                f"Epoch {epoch + 1}/{epochs} — "
                f"train loss : {avg_train_loss:.4f} | "
                f"val loss : {avg_val_loss:.4f}"
            )
        else:
            print(
                f"Epoch {epoch + 1}/{epochs} — "
                f"train loss : {avg_train_loss:.4f}"
            )

    return model, history