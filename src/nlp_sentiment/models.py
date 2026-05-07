"""PyTorch model architectures used across the project."""
from __future__ import annotations

import torch
import torch.nn as nn


class LogisticRegression(nn.Module):
    """Linear classifier on a fixed-size feature vector.

    Used for both n-gram count features and TF-IDF features. The model
    is intentionally simple so its coefficients remain interpretable.
    """

    def __init__(self, input_dim: int, output_dim: int = 2):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class BiLSTMClassifierV2(nn.Module):
    """Bidirectional LSTM classifier with configurable pooling.

    Embeddings can be initialized from pretrained vectors (e.g. GloVe).
    Supports last-state, mean and max pooling over the BiLSTM outputs.
    """

    VALID_POOLINGS = {"last", "mean", "max"}

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 100,
        hidden_dim: int = 256,
        num_layers: int = 2,
        output_dim: int = 2,
        dropout_rate: float = 0.3,
        lstm_dropout: float = 0.3,
        pooling: str = "mean",
        pretrained_embeddings: torch.Tensor | None = None,
        freeze_embeddings: bool = False,
    ):
        super().__init__()
        if pooling not in self.VALID_POOLINGS:
            raise ValueError(
                f"pooling must be one of {self.VALID_POOLINGS}, got {pooling!r}"
            )
        self.pooling = pooling

        # Embedding layer — initialisable depuis GloVe pré-entraîné
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False

        self.embedding_dropout = nn.Dropout(dropout_rate)

        # BiLSTM : on désactive le dropout interne quand num_layers=1
        # (PyTorch émet un warning sinon)
        effective_lstm_dropout = lstm_dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=effective_lstm_dropout,
        )

        # Sortie BiLSTM = 2 × hidden_dim (concat des deux directions)
        self.classifier_dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(2 * hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        embedded = self.embedding_dropout(embedded)

        # outputs : (batch, seq_len, 2*hidden_dim)
        outputs, (h_n, _) = self.lstm(embedded)

        if self.pooling == "last":
            # h_n : (num_layers*2, batch, hidden_dim)
            # On concat les états cachés de la dernière couche (forward + backward)
            forward_last = h_n[-2]
            backward_last = h_n[-1]
            pooled = torch.cat([forward_last, backward_last], dim=-1)
        elif self.pooling == "mean":
            pooled = outputs.mean(dim=1)
        else:
            pooled = outputs.max(dim=1).values

        return self.classifier(self.classifier_dropout(pooled))