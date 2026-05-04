# Analyse de l'ablation LSTM

## Contexte

Après avoir établi un baseline solide avec TF-IDF (val F1 = 91.92%), j'ai
exploré si un BiLSTM avec embeddings GloVe pouvait dépasser ce score.

J'ai procédé en deux phases :
1. **Sweep sur la taille du vocabulaire** (15k, 30k, 60k mots)
2. **Ablation contrôlée** sur l'architecture et les embeddings (5 runs)

Toutes les expériences ont été menées sur Colab GPU (T4) avec une graine
aléatoire fixe pour la reproductibilité.

## Résultats de l'ablation V2

| # | Configuration | Val F1 | Δ |
|---|---------------|--------|---|
| 0 | Baseline (hidden=64, 1 couche, last pool, GloVe 100d) | 0.8934 | — |
| 1 | + hidden=256 | 0.8930 | −0.0004 |
| 2 | + 2 couches LSTM | 0.8907 | −0.0023 |
| 3 | + mean pooling | 0.9025 | +0.0118 |
| 4 | + GloVe 300d | **0.9140** | +0.0115 |

## Interprétation

### Ce qui n'a pas marché

**Augmenter la capacité du modèle (hidden 256, 2 couches) n'a apporté
aucun gain mesurable.** Le hidden plus large a même légèrement dégradé
la performance (probablement par overfitting sur 22 000 exemples
d'entraînement). La 2e couche a aggravé le problème : la val loss à
l'epoch 10 atteignait 0.47 contre 0.05 en train, signe d'un fort
surapprentissage.

### Ce qui a marché

**Le mean pooling a apporté le saut le plus significatif (+1.2%).** Cette
modification n'ajoute aucun paramètre mais change radicalement la stratégie
d'extraction d'information : au lieu d'utiliser uniquement le dernier état
caché du LSTM (qui peut "oublier" le début de la critique), le mean pool
agrège l'information sur toute la séquence.

**GloVe 300d a ajouté +1.2% supplémentaire.** Plus de dimensions = plus de
signal sémantique. Le coût est un doublement des paramètres (5.1M → 10.9M)
mais Colab GPU le supporte sans problème.

### Le résultat final : LSTM = TF-IDF (statistiquement)

Le combo final (hidden=256, 2 couches, mean pool, GloVe 300d) atteint
**91.40% F1**, soit 0.52% en dessous du TF-IDF (91.92%). Cette différence
est dans la marge d'erreur statistique (±1.2% sur 3000 critiques), donc
les deux modèles sont **statistiquement équivalents**.

### Diagnostic d'overfitting

Les meilleurs modèles atteignent leur optimum à l'epoch 2-3, signe que
la capacité du modèle excède la complexité de la tâche pour cette taille
de dataset. Pour aller plus loin avec un LSTM, il faudrait :
- Soit plus de régularisation (dropout 0.5, weight decay)
- Soit plus de données (impossible sans changer de dataset)
- Soit changer de paradigme (transformers pré-entraînés sur des corpus
  massifs)

## Décision

Plutôt que de continuer à tuner le LSTM pour grappiller marginalement
contre TF-IDF, j'ai choisi d'explorer les transformers (DistilBERT) qui
représentent le vrai saut technologique sur cette tâche.

## Coût d'entraînement

- 9 runs LSTM au total (5 ablation + 3 sweep vocab + 1 baseline)
- Temps total Colab GPU T4 : ~50 minutes
- Tous les runs sont logués dans `outputs/experiments.json`