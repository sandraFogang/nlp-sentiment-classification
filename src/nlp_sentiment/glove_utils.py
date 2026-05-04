"""
glove_utils.py — Chargement de GloVe et initialisation des embeddings.

GloVe (Pennington et al., 2014) : embeddings pré-entraînés sur 6B mots
de Wikipedia + Gigaword. On utilise GloVe 6B.100d (100 dimensions).

Format des fichiers GloVe : un fichier texte où chaque ligne est :
    mot dim_1 dim_2 ... dim_100
"""
from pathlib import Path

import torch


def load_glove_embeddings(glove_path: Path, embedding_dim: int) -> dict[str, torch.Tensor]:
    """
    Charge GloVe depuis un fichier .txt.

    Args:
        glove_path: Chemin vers le fichier GloVe (ex: glove.6B.100d.txt).
        embedding_dim: Dimension attendue (100 pour GloVe 6B.100d).

    Returns:
        Dict {mot: tenseur(embedding_dim,)} pour environ 400 000 mots.
    """
    if not glove_path.exists():
        raise FileNotFoundError(
            f"Fichier GloVe introuvable : {glove_path}\n"
            f"Téléchargez-le depuis https://nlp.stanford.edu/projects/glove/"
        )

    print(f"Chargement de GloVe depuis {glove_path}...")
    embeddings_dict = {}

    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ")
            word = parts[0]
            vector = torch.tensor([float(x) for x in parts[1:]], dtype=torch.float32)

            if vector.shape[0] != embedding_dim:
                raise ValueError(
                    f"Dimension inattendue pour '{word}' : "
                    f"{vector.shape[0]} vs {embedding_dim}"
                )

            embeddings_dict[word] = vector

    print(f"  → {len(embeddings_dict)} mots chargés depuis GloVe.")
    return embeddings_dict


def build_embedding_matrix(
    vocab: dict[str, int],
    glove_embeddings: dict[str, torch.Tensor],
    embedding_dim: int,
) -> torch.Tensor:
    """
    Construit la matrice d'embeddings pour le vocabulaire IMDB.

    Pour chaque mot du vocabulaire :
    - S'il existe dans GloVe → on copie le vecteur GloVe
    - Sinon → on initialise aléatoirement (loi normale petite)

    Le token <PAD> (indice 0) reste à zéro (par convention).
    Le token <UNK> (indice 1) est initialisé à la moyenne de GloVe (robuste).

    Args:
        vocab: Vocabulaire IMDB {mot: indice}.
        glove_embeddings: Dict GloVe {mot: tenseur}.
        embedding_dim: Dimension (100 pour GloVe 6B.100d).

    Returns:
        Matrice (vocab_size, embedding_dim).
    """
    vocab_size = len(vocab)
    embedding_matrix = torch.zeros(vocab_size, embedding_dim)

    # Calcul de la moyenne de GloVe pour initialiser <UNK>
    glove_mean = torch.stack(list(glove_embeddings.values())).mean(dim=0)

    n_found = 0
    n_random = 0

    for word, idx in vocab.items():
        if idx == 0:  # <PAD>
            embedding_matrix[idx] = torch.zeros(embedding_dim)
        elif idx == 1:  # <UNK>
            embedding_matrix[idx] = glove_mean
        elif word in glove_embeddings:
            embedding_matrix[idx] = glove_embeddings[word]
            n_found += 1
        else:
            # Initialisation aléatoire (loi normale d'écart-type 0.1)
            embedding_matrix[idx] = torch.randn(embedding_dim) * 0.1
            n_random += 1

    coverage = n_found / (vocab_size - 2) * 100  # -2 pour PAD et UNK
    print(
        f"Couverture GloVe : {n_found}/{vocab_size - 2} mots "
        f"({coverage:.1f}%), {n_random} initialisés aléatoirement."
    )

    return embedding_matrix