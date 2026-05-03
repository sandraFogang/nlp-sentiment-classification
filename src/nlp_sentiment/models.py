"""
models.py — Architectures des classificateurs de sentiment.

Ce module définit deux modèles :
- LogisticRegression : classificateur linéaire utilisé pour les n-grammes
- LSTMClassifier : classificateur récurrent qui traite les séquences mot par mot
"""
import torch
import torch.nn as nn


class LogisticRegression(nn.Module):
    """
    Régression logistique multi-classes.

    Utilisée comme couche finale des classificateurs n-grammes
    et comme couche de classification du LSTM.

    Args:
        input_dim: Taille du vecteur d'entrée (taille du vocabulaire pour
                   les n-grammes, ou hidden_dim pour le LSTM).
        output_dim: Nombre de classes (2 dans notre cas : pos/neg).
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Note : on retourne les logits (sans softmax)
        # car CrossEntropyLoss applique elle-même le softmax
        return self.linear(x)


class LSTMClassifier(nn.Module):
    """
    Classificateur de critiques basé sur un LSTM.

    Architecture :
    1. Embedding : transforme chaque indice de mot en vecteur dense
    2. LSTM : traite la séquence de vecteurs mot par mot
    3. Couche de classification : transforme l'état caché final en logits

    Args:
        vocab_size: Taille du vocabulaire (incluant <PAD> et <UNK>).
        emb_dim: Dimension des vecteurs d'embedding.
        hidden_dim: Dimension de l'état caché du LSTM.
        output_dim: Nombre de classes (2 : pos/neg).
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hidden_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # padding_idx=0 : les tokens de padding ne contribuent pas au gradient
        self.embeddings = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTMCell(emb_dim, hidden_dim)
        self.final_classifier_layer = LogisticRegression(hidden_dim, output_dim)

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_seq: Tenseur de forme (batch_size, sequence_length)
                       contenant des indices de mots.

        Returns:
            Logits de classification de forme (batch_size, output_dim).
        """
        batch_size = input_seq.size(0)
        sequence_length = input_seq.size(1)

        # États cachés et de cellule initiaux (zéros)
        h_prior = torch.zeros(batch_size, self.hidden_dim, device=input_seq.device)
        c_prior = torch.zeros(batch_size, self.hidden_dim, device=input_seq.device)

        # On transpose pour itérer sur la dimension temporelle
        # (sequence_length, batch_size)
        input_t = torch.transpose(input_seq, 0, 1)

        # Boucle récurrente : un pas de LSTM par mot de la séquence
        for i in range(sequence_length):
            x_i = self.embeddings(input_t[i])
            h_i, c_i = self.lstm(x_i, (h_prior, c_prior))
            h_prior, c_prior = h_i, c_i

        # Seul le dernier état caché est utilisé pour la classification
        return self.final_classifier_layer(h_prior)