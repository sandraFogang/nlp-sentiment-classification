"""
test_predict.py — Tests d'intégration de la fonction predict.

Note : ces tests nécessitent qu'un modèle soit entraîné et présent
dans models/. Si le modèle n'existe pas, les tests sont automatiquement
skippés (pas en échec).

Pour entraîner le modèle :
    python scripts/train_champion_tfidf.py
"""
import pytest

from nlp_sentiment.config import MODEL_PATH, VOCAB_PATH
from nlp_sentiment.predict import predict


# Skip tous les tests si le modèle n'a pas été entraîné
pytestmark = pytest.mark.skipif(
    not (MODEL_PATH.exists() and VOCAB_PATH.exists()),
    reason="Modèle non entraîné — lancez `python scripts/train_champion_tfidf.py`",
)


def test_predict_returns_correct_format():
    """La fonction predict retourne un dict avec les bonnes clés."""
    result = predict("A wonderful and beautiful film with great acting.")
    assert "label" in result
    assert "confidence" in result
    assert "probabilities" in result
    assert "model_type" in result


def test_predict_label_is_valid():
    """Le label retourné est soit 'positif' soit 'négatif'."""
    result = predict("Great movie, loved every minute of it.")
    assert result["label"] in ["positif", "négatif"]


def test_predict_probabilities_sum_to_one():
    """Les probabilités somment à 1 (à 0.01 près à cause des arrondis float)."""
    result = predict("Average movie, nothing special.")
    total = result["probabilities"]["positif"] + result["probabilities"]["négatif"]
    assert abs(total - 1.0) < 0.01


def test_predict_empty_text_raises():
    """Un texte vide doit lever une ValueError."""
    with pytest.raises(ValueError):
        predict("")