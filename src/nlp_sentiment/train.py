"""
train.py — Fonctions d'entraînement des classificateurs.

Ce module fournit une fonction `train` générique qui fonctionne
pour les deux types de classificateurs (n-grammes et LSTM).
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from nlp_sentiment.config import DEVICE, LEARNING_RATE, N_EPOCHS


def train(
    model: nn.Module,
    dataloader: DataLoader,
    lr: float = LEARNING_RATE,
    epochs: int = N_EPOCHS,
    device: torch.device = DEVICE,
) -> nn.Module:
    """
    Entraîne un modèle de classification par descente de gradient.

    Args:
        model: Le modèle PyTorch à entraîner.
        dataloader: DataLoader contenant les données d'entraînement.
        lr: Taux d'apprentissage (learning rate).
        epochs: Nombre de passes complètes sur les données.
        device: 'cpu' ou 'cuda'.

    Returns:
        Le modèle entraîné (modifié sur place mais aussi retourné).
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(epochs):
        train_loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        epoch_loss = 0.0

        for batch_inputs, batch_labels, _ in train_loop:
            # Déplace les tenseurs sur le device (CPU ou GPU)
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)

            # CrossEntropyLoss attend des indices de classe, pas du one-hot
            batch_labels = batch_labels.argmax(dim=1)

            # Étapes classiques d'une boucle d'entraînement PyTorch :
            # 1. Mettre les gradients à zéro
            optimizer.zero_grad()
            # 2. Propagation avant (forward pass)
            logits = model(batch_inputs)
            # 3. Calcul de la perte
            loss = criterion(logits, batch_labels)
            # 4. Rétropropagation (backward pass)
            loss.backward()
            # 5. Mise à jour des poids
            optimizer.step()

            epoch_loss += loss.item()
            train_loop.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs} terminé — perte moyenne : {avg_loss:.4f}")

    return model