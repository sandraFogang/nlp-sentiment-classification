"""
config.py — Configuration centralisée du projet.

Toutes les constantes du projet (graines aléatoires, hyperparamètres,
chemins) sont définies ici pour faciliter la maintenance.
"""
from pathlib import Path

import torch

# === Reproductibilité ===
# Graines aléatoires identiques à chaque exécution
RANDOM_SEED = 202601
TORCH_SEED = 202401

# === Matériel ===
# Utilise le GPU si disponible, sinon le CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Données ===
# Proportion du corpus utilisée pour l'entraînement (le reste pour le test)
TRAIN_SPLIT = 0.8

# Liste des classes possibles
REVIEW_CLASSES = ["neg", "pos"]

# === Vocabulaire ===
# Taille maximale du vocabulaire de n-grammes
MAX_VOCAB_SIZE = 30_000

# === Hyperparamètres d'entraînement ===
BATCH_SIZE = 32
LEARNING_RATE = 0.001
N_EPOCHS = 5

# === Hyperparamètres du LSTM ===
LSTM_EMBEDDING_DIM = 64
LSTM_HIDDEN_DIM = 64

# === Chemins du projet ===
# Path(__file__) = ce fichier (config.py)
# .parent = le dossier qui le contient (nlp_sentiment/)
# .parent.parent = src/
# .parent.parent.parent = la racine du projet
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data"

# Chemin où le modèle entraîné sera sauvegardé
MODEL_PATH = MODELS_DIR / "classifier.pt"
VOCAB_PATH = MODELS_DIR / "vocab.pkl"