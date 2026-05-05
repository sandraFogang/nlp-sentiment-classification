"""
bert_data.py — Préparation des données pour DistilBERT.

Utilise le tokenizer pré-entraîné de Hugging Face qui :
- Fait la tokenisation WordPiece (sous-mots)
- Ajoute les tokens spéciaux [CLS] et [SEP]
- Gère le padding et la troncature
- Retourne aussi un attention_mask (1 pour les vrais tokens, 0 pour padding)
"""
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from nlp_sentiment.config import REVIEW_CLASSES


# Modèle par défaut : DistilBERT base (uncased = pas de distinction maj/min)
DEFAULT_BERT_MODEL = "distilbert-base-uncased"


def build_tokenizer(model_name: str = DEFAULT_BERT_MODEL):
    """
    Charge le tokenizer pré-entraîné depuis Hugging Face.

    La première fois, télécharge ~250 KB depuis huggingface.co.
    Les fois suivantes, utilise le cache local.
    """
    return AutoTokenizer.from_pretrained(model_name)


class BertReviewDataset(Dataset):
    """
    Dataset PyTorch utilisant le tokenizer DistilBERT.

    Chaque critique est représentée par :
    - input_ids : indices des tokens WordPiece (avec [CLS] et [SEP])
    - attention_mask : 1 pour les vrais tokens, 0 pour padding
    - label : one-hot (cohérent avec le reste du pipeline)
    """

    def __init__(
        self,
        reviews: list[tuple[str, str]],
        tokenizer,
        max_seq_len: int = 512,
    ):
        """
        Args:
            reviews: Liste de tuples (texte, classe) — 'pos' ou 'neg'.
            tokenizer: Tokenizer Hugging Face (chargé via build_tokenizer).
            max_seq_len: Longueur max après tokenisation (512 = max DistilBERT).
        """
        self.reviews = reviews
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.reviews)

    def __getitem__(self, idx: int) -> tuple[dict, torch.Tensor, int]:
        text, class_label = self.reviews[idx]

        # Tokenisation : retourne un dict avec input_ids et attention_mask
        encoded = self.tokenizer(
            text,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # encoded["input_ids"] : (1, max_seq_len) → on enlève la dimension batch
        feature = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }

        # Label one-hot (cohérent avec le pipeline existant)
        label = torch.zeros(len(REVIEW_CLASSES))
        label[REVIEW_CLASSES.index(class_label)] = 1

        return feature, label, idx


def prepare_bert_data(
    train_data: list[tuple[str, str]],
    val_data: list[tuple[str, str]],
    model_name: str = DEFAULT_BERT_MODEL,
    max_seq_len: int = 512,
) -> tuple[BertReviewDataset, BertReviewDataset, object]:
    """
    Pipeline complet : tokenizer + datasets train/val.

    Returns:
        Tuple (train_dataset, val_dataset, tokenizer).
    """
    print(f"Chargement du tokenizer {model_name}...")
    tokenizer = build_tokenizer(model_name)

    train_dataset = BertReviewDataset(train_data, tokenizer, max_seq_len=max_seq_len)
    val_dataset = BertReviewDataset(val_data, tokenizer, max_seq_len=max_seq_len)

    print(f"  → Datasets créés : train={len(train_dataset)}, val={len(val_dataset)}")
    return train_dataset, val_dataset, tokenizer