"""Text preprocessing utilities (tokenization, normalization, n-grams)."""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Iterable

import nltk

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


# Pattern de ponctuation en attribut module-level (évite de le recompiler à chaque appel)
_PUNCT_PATTERN = re.compile(f"[{re.escape(string.punctuation)}]")
# Le tag HTML <br /> apparaît partout dans le corpus IMDB
_HTML_BR_PATTERN = re.compile(r"<br\s*/?>")


def preprocess(text: str) -> list[str]:
    """Tokenize an English review with light normalization.

    Lowercases, strips HTML <br>, removes punctuation, and tokenizes.
    Used identically by every model so feature spaces stay aligned.

    Args:
        text: Raw review text.

    Returns:
        List of lowercase tokens.
    """
    if not text:
        return []

    text = text.lower()
    text = _HTML_BR_PATTERN.sub(" ", text)
    text = _PUNCT_PATTERN.sub(" ", text)

    # NLTK gère mieux les contractions et apostrophes que str.split()
    return nltk.word_tokenize(text)


def build_ngrams(tokens: list[str], n: int) -> list[tuple]:
    """Build n-gram tuples from a list of tokens.

    Args:
        tokens: Output of :func:`preprocess`.
        n: Size of the n-grams (1 = unigrams, 2 = bigrams, 3 = trigrams).

    Returns:
        List of n-gram tuples. Empty if len(tokens) < n.
    """
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def build_vocabulary(
    tokenized_corpus: Iterable[list[str]],
    n: int = 2,
    min_count: int = 5,
    max_vocab_size: int | None = None,
) -> list[tuple]:
    """Build a sorted n-gram vocabulary from a tokenized corpus.

    Args:
        tokenized_corpus: Iterable of tokenized reviews.
        n: N-gram size.
        min_count: Discard n-grams seen fewer than this many times.
        max_vocab_size: Cap the vocabulary size (top-K by frequency).

    Returns:
        List of n-gram tuples sorted by descending frequency.
    """
    counter: Counter = Counter()
    for tokens in tokenized_corpus:
        counter.update(build_ngrams(tokens, n))

    # Filtrage par fréquence minimale (seuil min_count)
    filtered = [(ng, c) for ng, c in counter.items() if c >= min_count]
    # Tri décroissant : les n-grams les plus fréquents d'abord
    filtered.sort(key=lambda x: -x[1])

    if max_vocab_size is not None:
        filtered = filtered[:max_vocab_size]

    return [ng for ng, _ in filtered]