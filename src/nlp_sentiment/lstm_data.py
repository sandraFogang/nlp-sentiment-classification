"""
lstm_data.py — Préparation des données pour le LSTM.

Différences clés par rapport au pipeline n-grammes/TF-IDF :
- Le vocabulaire est limité (top-K mots) car chaque mot = 1 embedding
- Les critiques sont des SÉQUENCES d'indices (pas des vecteurs sparses)
- Padding : toutes les séquences ont la même longueur (MAX_SEQ_LEN)
- Tokens spéciaux : <PAD>=0 (remplissage), <UNK>=1 (mot inconnu)
"""
from collections import Counter

import torch
from torch.utils.data import Dataset

from nlp_sentiment.config import REVIEW_CLASSES
from nlp_sentiment.preprocessor import preprocess


# === Tokens spéciaux ===
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1


def build_word_vocab(
    tokenized_reviews: list[tuple[list[str], str]],
    max_vocab_size: int = 30000,
    min_count: int = 5,
) -> dict[str, int]:
    """
    Construit un vocabulaire mot -> indice pour le LSTM.

    Garde les `max_vocab_size` mots les plus fréquents qui apparaissent
    au moins `min_count` fois dans le train.

    Args:
        tokenized_reviews: Liste de (tokens, classe) du train.
        max_vocab_size: Taille max du vocabulaire (incluant <PAD> et <UNK>).
        min_count: Fréquence minimale d'un mot.

    Returns:
        Dictionnaire {mot: indice} où :
        - PAD_IDX (0) → <PAD>
        - UNK_IDX (1) → <UNK>
        - 2+ → mots du corpus
    """
    counter = Counter()
    for tokens, _ in tokenized_reviews:
        counter.update(tokens)

    # Filtrage par fréquence et taille max (-2 pour PAD et UNK)
    most_common = [
        (word, count)
        for word, count in counter.most_common(max_vocab_size - 2)
        if count >= min_count
    ]

    vocab = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    for word, _ in most_common:
        vocab[word] = len(vocab)

    return vocab


def encode_review(
    tokens: list[str],
    vocab: dict[str, int],
    max_seq_len: int,
) -> list[int]:
    """
    Convertit une critique tokenisée en séquence d'indices.

    - Remplace chaque mot par son indice dans le vocabulaire (UNK_IDX si absent).
    - Tronque à max_seq_len si trop long.
    - Padde avec PAD_IDX si trop court.

    Args:
        tokens: Liste de mots tokenisés.
        vocab: Vocabulaire mot -> indice.
        max_seq_len: Longueur cible de la séquence.

    Returns:
        Liste de max_seq_len entiers.
    """
    indices = [vocab.get(token, UNK_IDX) for token in tokens]

    # Troncature si trop long
    if len(indices) > max_seq_len:
        indices = indices[:max_seq_len]
    # Padding si trop court
    else:
        indices = indices + [PAD_IDX] * (max_seq_len - len(indices))

    return indices


class LSTMSequenceDataset(Dataset):
    """
    Dataset PyTorch pour le LSTM : séquences d'indices de longueur fixe.

    Chaque critique est représentée par un tenseur d'entiers de taille
    (max_seq_len,), prêt à être consommé par un LSTM PyTorch.
    """

    def __init__(
        self,
        tokenized_reviews: list[tuple[list[str], str]],
        vocab: dict[str, int],
        max_seq_len: int = 512,
    ):
        self.tokenized_reviews = tokenized_reviews
        self.vocab = vocab
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.tokenized_reviews)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        tokens, class_label = self.tokenized_reviews[idx]

        # Encodage en séquence d'indices
        indices = encode_review(tokens, self.vocab, self.max_seq_len)
        feature = torch.tensor(indices, dtype=torch.long)

        # Label one-hot (cohérent avec le reste du pipeline)
        label = torch.zeros(len(REVIEW_CLASSES))
        label[REVIEW_CLASSES.index(class_label)] = 1

        return feature, label, idx


def prepare_lstm_data(
    train_data: list[tuple[str, str]],
    val_data: list[tuple[str, str]],
    max_vocab_size: int = 30000,
    min_count: int = 5,
    max_seq_len: int = 512,
) -> tuple[LSTMSequenceDataset, LSTMSequenceDataset, dict[str, int]]:
    """
    Pipeline complet de préparation des données LSTM.

    1. Tokenise train + val (avec preprocess())
    2. Construit le vocabulaire à partir du train uniquement
    3. Crée les datasets PyTorch

    Args:
        train_data: Liste de (texte, classe) du train.
        val_data: Liste de (texte, classe) du val.
        max_vocab_size: Taille max du vocabulaire.
        min_count: Fréquence minimale d'un mot.
        max_seq_len: Longueur des séquences (padding/troncature).

    Returns:
        Tuple (train_dataset, val_dataset, vocab).
    """
    print("Tokenisation du train et du val...")
    tokenized_train = [(preprocess(text), label) for text, label in train_data]
    tokenized_val = [(preprocess(text), label) for text, label in val_data]

    print(f"Construction du vocabulaire (max={max_vocab_size}, min_count={min_count})...")
    vocab = build_word_vocab(
        tokenized_train,
        max_vocab_size=max_vocab_size,
        min_count=min_count,
    )
    print(f"  → {len(vocab)} mots dans le vocabulaire (incluant <PAD> et <UNK>).")

    train_dataset = LSTMSequenceDataset(tokenized_train, vocab, max_seq_len=max_seq_len)
    val_dataset = LSTMSequenceDataset(tokenized_val, vocab, max_seq_len=max_seq_len)

    return train_dataset, val_dataset, vocab