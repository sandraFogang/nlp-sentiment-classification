"""
data.py — Chargement et préparation du corpus Movie Reviews.

Le corpus NLTK Movie Reviews contient 2 000 critiques de films
en anglais (1 000 positives, 1 000 négatives).
"""
import random

import nltk
from nltk.corpus import movie_reviews

from nlp_sentiment.config import RANDOM_SEED, TRAIN_SPLIT


def download_corpus_if_needed() -> None:
    """
    Télécharge le corpus NLTK Movie Reviews s'il n'est pas déjà présent.

    Cette fonction est silencieuse si le corpus est déjà téléchargé
    sur la machine.
    """
    try:
        movie_reviews.fileids()
    except LookupError:
        print("Téléchargement du corpus NLTK Movie Reviews...")
        nltk.download("movie_reviews", quiet=True)
        print("Corpus téléchargé avec succès.")


def load_movie_reviews() -> list[tuple[str, str]]:
    """
    Charge toutes les critiques de films depuis le corpus NLTK.

    Returns:
        Liste de tuples (texte_critique, classe) où classe est 'pos' ou 'neg'.
    """
    download_corpus_if_needed()

    documents = []
    for category in movie_reviews.categories():
        for fileid in movie_reviews.fileids(category):
            text = movie_reviews.raw(fileid)
            documents.append((text, category))

    return documents


def split_train_test(
    documents: list[tuple[str, str]],
    train_ratio: float = TRAIN_SPLIT,
    seed: int = RANDOM_SEED,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """
    Mélange les documents et les sépare en ensembles d'entraînement et de test.

    Args:
        documents: Liste de tuples (texte, classe).
        train_ratio: Proportion utilisée pour l'entraînement (0.8 par défaut).
        seed: Graine aléatoire pour la reproductibilité.

    Returns:
        Tuple (train_dataset, test_dataset).
    """
    random.seed(seed)
    shuffled = documents.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)
    train_dataset = shuffled[:split_idx]
    test_dataset = shuffled[split_idx:]

    return train_dataset, test_dataset