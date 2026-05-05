"""
find_demo_examples.py — Sélectionne des exemples IMDB pour la démo Streamlit.

Cherche dans le test set des critiques avec différents niveaux de confiance :
- Très haute confiance positive (>95%)
- Très haute confiance négative (>95%)
- Confiance modérée (60-75%)
- Erreur du modèle (label vrai différent de la prédiction)

Affiche les 3 premières critiques de chaque catégorie qui font moins de 800 caractères
(pour ne pas surcharger l'interface Streamlit).
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nlp_sentiment.data import load_imdb_splits
from nlp_sentiment.predict import predict


def main() -> None:
    print("Chargement du test set IMDB...")
    _, _, test_data = load_imdb_splits()

    # On limite à 500 exemples pour aller vite (déjà beaucoup)
    print("Analyse des 500 premières critiques du test set...\n")

    high_pos = []      # >95% positif
    high_neg = []      # >95% négatif
    moderate = []      # 60-75% (modèle hésite)
    errors = []        # le modèle se trompe

    # Le test set IMDB est ordonné (negs d'abord, puis pos)
    # On scanne plus large pour avoir des exemples des deux classes
    sample = list(test_data[:500]) + list(test_data[12500:13000])
    for i, (text, true_label) in enumerate(sample):
        if len(text) > 800:
            continue

        result = predict(text)
        confidence = result["confidence"]
        predicted = "pos" if result["label"] == "positif" else "neg"

        if predicted != true_label:
            errors.append((text, true_label, predicted, confidence))
        elif result["label"] == "positif" and confidence > 0.95:
            high_pos.append((text, confidence))
        elif result["label"] == "négatif" and confidence > 0.95:
            high_neg.append((text, confidence))
        elif 0.55 < confidence < 0.75:
            moderate.append((text, result["label"], confidence))

    def display(category: str, items: list, n: int = 3) -> None:
        print("=" * 70)
        print(f"  {category}")
        print("=" * 70)
        for i, item in enumerate(items[:n], 1):
            if len(item) == 2:
                text, conf = item
                print(f"\n[{i}] Confiance : {conf:.1%}")
            elif len(item) == 3:
                text, label, conf = item
                print(f"\n[{i}] Prédiction : {label} (confiance {conf:.1%})")
            else:
                text, true_label, pred, conf = item
                print(f"\n[{i}] Vrai : {true_label} | Prédit : {pred} ({conf:.1%})")
            print(f"    Longueur : {len(text)} caractères")
            print(f"    Texte : {text[:300]}...")
        print()

    display("Critiques très positives (>95% confiance)", high_pos)
    display("Critiques très négatives (>95% confiance)", high_neg)
    display("Critiques où le modèle hésite (55-75%)", moderate)
    display("Critiques où le modèle se trompe (mauvaise prédiction)", errors)


if __name__ == "__main__":
    main()
    