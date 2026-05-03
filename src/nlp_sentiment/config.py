"""
config.py — Configuration centralisée du projet.

Toutes les constantes (graines aléatoires, hyperparamètres, chemins)
sont définies ici pour faciliter la maintenance et la reproductibilité.
"""
from pathlib import Path

import torch

# === Reproductibilité ===
RANDOM_SEED = 202601
TORCH_SEED = 202401

# === Matériel ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Dataset ===
# IMDB Large Movie Review Dataset (Maas et al. 2011) — standard académique
DATASET_NAME = "imdb"

# Taille du val set extrait du train officiel (25 000 critiques)
# 3 000 = équilibre entre statistique fiable et train assez grand pour le DL
VAL_SIZE = 3000

# Liste des classes possibles (ordre important : index 0 = neg, 1 = pos)
REVIEW_CLASSES = ["neg", "pos"]

# === Vocabulaire ===
MAX_VOCAB_SIZE = 30_000

# === Hyperparamètres d'entraînement ===
BATCH_SIZE = 32
LEARNING_RATE = 0.001
N_EPOCHS = 5

# === Hyperparamètres du LSTM ===
LSTM_EMBEDDING_DIM = 64
LSTM_HIDDEN_DIM = 64

# === Chemins du projet ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data"

# Cache pour les datasets téléchargés (Hugging Face datasets)
HF_CACHE_DIR = DATA_DIR / "huggingface_cache"

# Chemins de sauvegarde du modèle de production (bigramme déployé)
MODEL_PATH = MODELS_DIR / "classifier.pt"
VOCAB_PATH = MODELS_DIR / "vocab.pkl"

# Chemin du fichier de tracking des expériences
EXPERIMENTS_PATH = OUTPUTS_DIR / "experiments.json"