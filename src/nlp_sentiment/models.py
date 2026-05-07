"""PyTorch model architectures used across the project."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.dropout_rate = dropout_rate

        # Note : nom 'embeddings' (avec s) pour aligner avec le state dict du checkpoint
        self.embeddings = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embeddings.weight.data.copy_(pretrained_embeddings)
            if freeze_embeddings:
                self.embeddings.weight.requires_grad = False

        # PyTorch émet un warning si dropout != 0 avec num_layers=1
        effective_lstm_dropout = lstm_dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=effective_lstm_dropout,
        )

        # Note : nom 'fc' pour aligner avec le state dict du checkpoint
        # Sortie BiLSTM = 2 × hidden_dim (concat forward + backward)
        self.fc = nn.Linear(2 * hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embeddings(x)
        embedded = F.dropout(embedded, p=self.dropout_rate, training=self.training)

        # outputs : (batch, seq_len, 2*hidden_dim)
        outputs, (h_n, _) = self.lstm(embedded)

        if self.pooling == "last":
            # h_n : (num_layers*2, batch, hidden_dim)
            # Concat des états cachés de la dernière couche (forward + backward)
            forward_last = h_n[-2]
            backward_last = h_n[-1]
            pooled = torch.cat([forward_last, backward_last], dim=-1)
        elif self.pooling == "mean":
            pooled = outputs.mean(dim=1)
        else:
            pooled = outputs.max(dim=1).values

        pooled = F.dropout(pooled, p=self.dropout_rate, training=self.training)
        return self.fc(pooled)