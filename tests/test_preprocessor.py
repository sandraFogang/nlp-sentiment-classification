"""
test_preprocessor.py — Tests unitaires de la fonction preprocess.

Lancer les tests :
    pytest
ou
    pytest tests/test_preprocessor.py -v
"""
from nlp_sentiment.preprocessor import preprocess


def test_preprocess_lowercase():
    """Le preprocessing met le texte en minuscules."""
    result = preprocess("HELLO World")
    assert result == ["hello", "world"]


def test_preprocess_removes_punctuation():
    """La ponctuation est retirée."""
    result = preprocess("Wow, this is great!")
    assert result == ["wow", "this", "is", "great"]


def test_preprocess_empty_string():
    """Une chaîne vide retourne une liste vide."""
    assert preprocess("") == []


def test_preprocess_only_punctuation():
    """Une chaîne contenant uniquement de la ponctuation retourne une liste vide."""
    assert preprocess("!!! ??? ...") == []