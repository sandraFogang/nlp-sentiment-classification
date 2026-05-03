# Données — NLTK Movie Reviews

Ce dossier ne contient pas de fichiers de données. Le corpus utilisé est **NLTK Movie Reviews** (Pang & Lee, 2004), téléchargé directement via la bibliothèque NLTK au moment de l'exécution.

## Description du corpus

- 2 000 critiques de films en anglais
- 1 000 positives + 1 000 négatives (corpus équilibré)
- Source : Internet Movie Database (IMDB), prétraitées par Pang & Lee

## Téléchargement automatique

Au premier lancement de l'entraînement, NLTK télécharge le corpus dans le cache utilisateur :

```python
import nltk
nltk.download('movie_reviews')
```

## Référence

Pang, B., & Lee, L. (2004). *A Sentimental Education: Sentiment Analysis Using Subjectivity Summarization Based on Minimum Cuts*. Proceedings of ACL.