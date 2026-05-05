"""
bert_model.py — Classificateur de sentiment basé sur DistilBERT.

Architecture :
1. DistilBERT pré-entraîné (66M paramètres) → représentations contextuelles
2. Token [CLS] (premier token) → vecteur 768-dim qui résume la séquence
3. Dropout
4. Linear (768 → 2) → logits de classification

Modes :
- freeze_bert=True  : seul le classifier final est entraîné
- freeze_bert=False : tout DistilBERT + le classifier sont fine-tunés (recommandé)
"""
import torch
import torch.nn as nn
from transformers import AutoModel


DEFAULT_BERT_MODEL = "distilbert-base-uncased"


class BertSentimentClassifier(nn.Module):
    """
    Wrapper DistilBERT pour la classification binaire de sentiment.

    Utilise le vecteur du token [CLS] (premier token) comme représentation
    de la séquence — standard en classification BERT.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_BERT_MODEL,
        output_dim: int = 2,
        dropout_rate: float = 0.3,
        freeze_bert: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.freeze_bert = freeze_bert

        # Chargement du modèle pré-entraîné depuis Hugging Face
        self.bert = AutoModel.from_pretrained(model_name)

        # Geler ou non les paramètres de DistilBERT
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Dimension de sortie de DistilBERT (768 pour distilbert-base)
        hidden_size = self.bert.config.hidden_size

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, output_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len) indices de tokens.
            attention_mask: (batch, seq_len) 1 pour vrais tokens, 0 pour padding.

        Returns:
            Logits (batch, output_dim).
        """
        # DistilBERT retourne (batch, seq_len, hidden_size)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        # Token [CLS] = premier token = position 0
        cls_representation = sequence_output[:, 0, :]  # (batch, hidden_size)

        cls_representation = self.dropout(cls_representation)
        logits = self.classifier(cls_representation)
        return logits

    def count_trainable_params(self) -> int:
        """Compte les paramètres entraînables (utile pour comparer frozen vs full)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)