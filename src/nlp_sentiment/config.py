"""Project-wide configuration: paths, device, random seeds, label scheme."""
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Chemins clés du projet
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# Modèle de production (TF-IDF) servi par Streamlit en mode local
MODEL_PATH = MODELS_DIR / "classifier.pt"
VOCAB_PATH = MODELS_DIR / "vocab.pkl"

# Convention de classes : 0 = négatif, 1 = positif (alphabétique)
REVIEW_CLASSES = ["neg", "pos"]

# Graines pour reproductibilité (séparées pour numpy/random et pour PyTorch)
RANDOM_SEED = 202601
TORCH_SEED = 202401

# Device : GPU si disponible, sinon CPU (HF Spaces et Streamlit Cloud sont CPU-only)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Découpage des données (les valeurs exactes dépendent du dataset chargé)
TRAIN_SIZE = 22_000
VAL_SIZE = 3_000
TEST_SIZE = 25_000