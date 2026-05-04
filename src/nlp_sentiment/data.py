"""
data.py — Chargement et préparation du corpus IMDB.

Source : IMDB Large Movie Review Dataset (Maas et al. 2011)
- 25 000 critiques pour le train (50% pos, 50% neg)
- 25 000 critiques pour le test (50% pos, 50% neg)

Le val set est extrait du train officiel pour préserver l'intégrité
du test set (intouché jusqu'à l'évaluation finale).
"""
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from nlp_sentiment.config import (
    DATASET_NAME,
    HF_CACHE_DIR,
    RANDOM_SEED,
    REVIEW_CLASSES,
    VAL_SIZE,
)


def _label_int_to_str(label: int) -> str:
    """Convertit un label IMDB (0 ou 1) en chaîne ('neg' ou 'pos')."""
    return REVIEW_CLASSES[label]


def load_imdb_splits() -> tuple[
    list[tuple[str, str]],
    list[tuple[str, str]],
    list[tuple[str, str]],
]:
    """
    Charge IMDB 50k et retourne 3 splits : train, val, test.

    Le test set est le test officiel IMDB (25 000 critiques), intouché.
    Le val set (3 000 critiques) est extrait du train officiel de manière
    stratifiée et reproductible.

    Returns:
        Tuple (train, val, test) où chaque ensemble est une liste de
        tuples (texte, classe). La classe est 'pos' ou 'neg'.

    Note:
        Au premier appel, télécharge IMDB depuis Hugging Face (~80 Mo).
        Les appels suivants utilisent le cache local dans data/huggingface_cache.
    """
    # Crée le dossier de cache si nécessaire
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Télécharge (ou charge depuis le cache) le dataset complet
    dataset = load_dataset(DATASET_NAME, cache_dir=str(HF_CACHE_DIR))

    # === Test set : tel quel, on n'y touche pas ===
    test_texts = list(dataset["test"]["text"])
    test_labels = list(dataset["test"]["label"])
    test = [
        (text, _label_int_to_str(label))
        for text, label in zip(test_texts, test_labels)
    ]

    # === Train + Val : on split le train officiel ===
    # IMPORTANT : on convertit en listes Python natives AVANT le split
    # (les versions récentes de `datasets` ne supportent pas les indices numpy)
    train_texts_full = list(dataset["train"]["text"])
    train_labels_full = list(dataset["train"]["label"])

    # Split stratifié reproductible
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts_full,
        train_labels_full,
        test_size=VAL_SIZE,
        random_state=RANDOM_SEED,
        stratify=train_labels_full,
    )
    train = [
        (text, _label_int_to_str(label))
        for text, label in zip(train_texts, train_labels)
    ]
    val = [
        (text, _label_int_to_str(label))
        for text, label in zip(val_texts, val_labels)
    ]

    return train, val, test


def describe_splits(
    train: list[tuple[str, str]],
    val: list[tuple[str, str]],
    test: list[tuple[str, str]],
) -> None:
    """
    Affiche un résumé descriptif des 3 splits.

    Utile pour vérifier que le chargement et le split sont cohérents.
    """
    def class_distribution(dataset: list[tuple[str, str]]) -> dict[str, int]:
        return {
            cls: sum(1 for _, label in dataset if label == cls)
            for cls in REVIEW_CLASSES
        }

    print(f"{'Split':<10} {'Total':<8} {'neg':<6} {'pos':<6}")
    print("-" * 32)
    for name, dataset in [("Train", train), ("Val", val), ("Test", test)]:
        dist = class_distribution(dataset)
        print(f"{name:<10} {len(dataset):<8} {dist['neg']:<6} {dist['pos']:<6}")