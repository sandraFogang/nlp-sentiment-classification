"""
preprocessor.py — Tokenisation, vocabulaires n-grammes et Datasets PyTorch.

Ce module contient :
- La fonction de prétraitement du texte (tokenisation simple)
- La construction de vocabulaires de n-grammes (uni/bi/tri)
- Les classes Dataset pour les classificateurs n-grammes et LSTM
"""
import re
import string
from collections import Counter

import torch
from torch.utils.data import Dataset

from nlp_sentiment.config import REVIEW_CLASSES


def preprocess(review: str) -> list[str]:
    """
    Convertit une critique brute en liste de tokens.

    Étapes :
    1. Met le texte en minuscules
    2. Retire la ponctuation
    3. Sépare en mots (tokenisation simple par espaces)

    Args:
        review: Texte brut d'une critique.

    Returns:
        Liste de tokens (mots) en minuscules sans ponctuation.
    """
    review = review.lower()
    # Retire la ponctuation
    review = re.sub(f"[{re.escape(string.punctuation)}]", " ", review)
    # Sépare en tokens et filtre les chaînes vides
    tokens = [t for t in review.split() if t]
    return tokens


def build_ngram_vocab(
    tokenized_reviews: list[tuple[list[str], str]],
    n: int,
    max_size: int,
) -> list[tuple]:
    """
    Construit un vocabulaire de n-grammes à partir d'un corpus tokenisé.

    Garde uniquement les `max_size` n-grammes les plus fréquents.

    Args:
        tokenized_reviews: Liste de tuples (tokens, classe).
        n: Taille des n-grammes (1=unigrammes, 2=bigrammes, 3=trigrammes).
        max_size: Nombre maximum de n-grammes à garder.

    Returns:
        Liste des n-grammes les plus fréquents (sous forme de tuples).
    """
    counter = Counter()
    for tokens, _ in tokenized_reviews:
        # Génère les n-grammes en glissant une fenêtre sur les tokens
        ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        counter.update(ngrams)

    # Garde les `max_size` n-grammes les plus fréquents
    most_common = counter.most_common(max_size)
    return [ngram for ngram, _ in most_common]

def build_ngram_vocab_min_count(
    tokenized_reviews: list[tuple[list[str], str]],
    n: int,
    min_count: int,
) -> list[tuple]:
    """
    Construit un vocabulaire de n-grammes par seuil de fréquence minimale.

    Garde tous les n-grammes apparaissant au moins `min_count` fois dans
    le corpus, sans limite supérieure sur la taille du vocabulaire.

    Cette stratégie est souvent préférable à `build_ngram_vocab` (top-K)
    car elle élimine le bruit statistique des n-grammes ultra-rares
    sans imposer de seuil arbitraire sur la taille.

    Args:
        tokenized_reviews: Liste de tuples (tokens, classe).
        n: Taille des n-grammes.
        min_count: Fréquence minimale requise (un n-gramme vu < min_count fois
                   dans le corpus est exclu).

    Returns:
        Liste des n-grammes retenus, triés par fréquence décroissante.
    """
    counter = Counter()
    for tokens, _ in tokenized_reviews:
        ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        counter.update(ngrams)

    # Garde tous les n-grammes vus au moins min_count fois
    filtered = [(ngram, count) for ngram, count in counter.items() if count >= min_count]
    # Trie par fréquence décroissante (cohérent avec build_ngram_vocab)
    filtered.sort(key=lambda x: x[1], reverse=True)

    return [ngram for ngram, _ in filtered]

def build_lstm_vocab(
    tokenized_reviews: list[tuple[list[str], str]],
    max_size: int,
) -> dict[str, int]:
    """
    Construit un vocabulaire mot -> indice pour le LSTM.

    L'indice 0 est réservé au token de padding.
    L'indice 1 est réservé aux mots inconnus (UNK).

    Args:
        tokenized_reviews: Liste de tuples (tokens, classe).
        max_size: Nombre maximum de mots à garder.

    Returns:
        Dictionnaire {mot: indice}.
    """
    counter = Counter()
    for tokens, _ in tokenized_reviews:
        counter.update(tokens)

    most_common = counter.most_common(max_size - 2)  # -2 pour PAD et UNK
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in most_common:
        vocab[word] = len(vocab)

    return vocab


def class_to_onehot(class_label: str) -> torch.Tensor:
    """
    Convertit une étiquette de classe ('pos' ou 'neg') en vecteur one-hot.

    Args:
        class_label: 'pos' ou 'neg'.

    Returns:
        Tenseur de taille (2,) : [1, 0] pour 'neg', [0, 1] pour 'pos'.
    """
    onehot = torch.zeros(len(REVIEW_CLASSES))
    onehot[REVIEW_CLASSES.index(class_label)] = 1
    return onehot


class NgramReviewDataset(Dataset):
    """
    Dataset PyTorch pour les classificateurs n-grammes.

    Chaque critique est représentée par un vecteur de comptage :
    chaque dimension correspond à un n-gramme du vocabulaire.
    """

    def __init__(
        self,
        tokenized_reviews: list[tuple[list[str], str]],
        ngram_vocab: list[tuple],
        n: int,
    ):
        self.tokenized_reviews = tokenized_reviews
        self.ngram_vocab = ngram_vocab
        self.n = n
        # Créé un dictionnaire {ngram: indice} pour des lookups rapides
        self.ngram_to_idx = {ngram: i for i, ngram in enumerate(ngram_vocab)}

    def __len__(self) -> int:
        return len(self.tokenized_reviews)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        tokens, class_label = self.tokenized_reviews[idx]

        # Vecteur de comptage des n-grammes
        feature = torch.zeros(len(self.ngram_vocab))
        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i : i + self.n])
            if ngram in self.ngram_to_idx:
                feature[self.ngram_to_idx[ngram]] += 1

        label = class_to_onehot(class_label)
        return feature, label, idx


class LSTMReviewDataset(Dataset):
    """
    Dataset PyTorch pour le classificateur LSTM.

    Chaque critique est représentée par une séquence d'indices de mots.
    """

    def __init__(
        self,
        tokenized_reviews: list[tuple[list[str], str]],
        vocab: dict[str, int],
    ):
        self.tokenized_reviews = tokenized_reviews
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.tokenized_reviews)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        tokens, class_label = self.tokenized_reviews[idx]

        # Convertit les tokens en indices, avec <UNK> pour les mots inconnus
        indices = [self.vocab.get(t, self.vocab["<UNK>"]) for t in tokens]
        feature = torch.tensor(indices, dtype=torch.long)

        label = class_to_onehot(class_label)
        return feature, label, idx