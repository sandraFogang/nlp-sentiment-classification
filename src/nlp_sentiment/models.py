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

class BiLSTMClassifier(nn.Module):
    """
    Classificateur bidirectionnel LSTM avec dropout et embedding initialisable.

    Architecture :
    1. Embedding (vocab_size → emb_dim) — initialisable avec GloVe
    2. Dropout (sur les embeddings, contre l'overfitting)
    3. LSTM bidirectionnel (1 couche par défaut)
    4. Dropout (sur l'état caché final)
    5. Linear (2 * hidden_dim → output_dim)

    Args:
        vocab_size: Taille du vocabulaire (incluant <PAD> et <UNK>).
        emb_dim: Dimension des embeddings (typiquement 100 si GloVe 6B.100d).
        hidden_dim: Dimension de l'état caché du LSTM (par direction).
        output_dim: Nombre de classes (2 : pos/neg).
        dropout_rate: Taux de dropout (0.3 typique).
        pretrained_embeddings: Tenseur (vocab_size, emb_dim) pour init GloVe (optionnel).
        freeze_embeddings: Si True, les embeddings ne sont PAS entraînés (rare).
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 100,
        hidden_dim: int = 64,
        output_dim: int = 2,
        dropout_rate: float = 0.3,
        pretrained_embeddings: torch.Tensor | None = None,
        freeze_embeddings: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # === Embedding layer ===
        # padding_idx=0 : les positions PAD ne contribuent pas au gradient
        self.embeddings = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

        # Initialisation pré-entraînée (GloVe) si fournie
        if pretrained_embeddings is not None:
            assert pretrained_embeddings.shape == (vocab_size, emb_dim), (
                f"Shape mismatch : {pretrained_embeddings.shape} vs ({vocab_size}, {emb_dim})"
            )
            self.embeddings.weight.data.copy_(pretrained_embeddings)
            if freeze_embeddings:
                self.embeddings.weight.requires_grad = False

        # === Dropout sur les embeddings ===
        self.embedding_dropout = nn.Dropout(dropout_rate)

        # === LSTM bidirectionnel ===
        # batch_first=True : input shape (batch, seq, features) au lieu de (seq, batch, features)
        # bidirectional=True : double la dimension de sortie (concat des 2 directions)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        # === Dropout avant la couche finale ===
        self.output_dropout = nn.Dropout(dropout_rate)

        # === Couche de classification ===
        # 2 * hidden_dim car bidirectionnel (concat forward + backward)
        self.fc = nn.Linear(2 * hidden_dim, output_dim)

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_seq: Tenseur (batch_size, seq_len) d'indices de mots.

        Returns:
            Logits (batch_size, output_dim).
        """
        # (batch, seq) → (batch, seq, emb_dim)
        embedded = self.embeddings(input_seq)
        embedded = self.embedding_dropout(embedded)

        # LSTM bidirectionnel
        # output : (batch, seq, 2 * hidden_dim)
        # h_n   : (2, batch, hidden_dim)  — états cachés finaux des 2 directions
        # c_n   : (2, batch, hidden_dim)
        _, (h_n, _) = self.lstm(embedded)

        # Concaténation des derniers états cachés des 2 directions
        # h_n[0] = forward,  h_n[1] = backward
        # → (batch, 2 * hidden_dim)
        h_concat = torch.cat([h_n[0], h_n[1]], dim=1)

        # Dropout + classification finale
        h_concat = self.output_dropout(h_concat)
        logits = self.fc(h_concat)

        return logits