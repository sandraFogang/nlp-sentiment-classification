"""Application web d'analyse de sentiment sur des critiques de films."""
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


st.set_page_config(
    page_title="Analyse de sentiment — Critiques de films",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ============================================================================
# Modèle (chargement avec fallback intelligent)
# ============================================================================
@st.cache_resource(show_spinner="Chargement du modèle...")
def _ensure_model_available() -> None:
    from nlp_sentiment.config import MODEL_PATH, VOCAB_PATH

    if not (MODEL_PATH.exists() and VOCAB_PATH.exists()):
        st.info("Premier démarrage : entraînement du modèle (environ 3 minutes).")
        from train_champion_tfidf import main as train_champion
        train_champion()


@st.cache_resource
def get_predict_fn():
    _ensure_model_available()
    from nlp_sentiment.predict import predict
    return predict


# ============================================================================
# Exemples du test set IMDB (jamais vus pendant l'entraînement)
# ============================================================================
EXAMPLES = {
    "clear_positive": {
        "label": "Critique positive — cas clair",
        "category": "Cas clairs (haute confiance attendue)",
        "text": (
            "Previous reviewer Claudio Carvalho gave a much better recap of the film's "
            "plot details than I could. What I recall mostly is that it was just so "
            "beautiful, in every sense - emotionally, visually, editorially - just gorgeous. "
            "If you like movies that are wonderful to look at, and also have emotional "
            "content to which that beauty is relevant, I think you will be glad to have "
            "seen this extraordinary and unusual work of art."
        ),
        "expected": "Sentiment réel : positif. Le modèle devrait être très confiant.",
        "is_tricky": False,
    },
    "clear_negative": {
        "label": "Critique négative — cas clair",
        "category": "Cas clairs (haute confiance attendue)",
        "text": (
            "Technically I'am a Van Damme Fan, or I was. This movie is so bad that I hated "
            "myself for wasting those 90 minutes. Do not let the name Isaac Florentine "
            "(Undisputed II) fool you, I had big hopes for this one, depending on what I "
            "saw in (Undisputed II), man.. was I wrong ??! All action fans wanted a better "
            "Van Damme movie. The story is weak, the fights are mediocre and the directing "
            "is below average. A complete disappointment."
        ),
        "expected": "Sentiment réel : négatif. Le modèle devrait être très confiant.",
        "is_tricky": False,
    },
    "moderate_negative": {
        "label": "Critique courte ambiguë",
        "category": "Cas nuancés (le modèle hésite légitimement)",
        "text": (
            "An obscure horror show filmed in the Everglades. Two couples stay overnight "
            "in a cabin after being made a little uneasy by the unfriendliness of the "
            "locals. Who, or what, are the Blood Stalkers? After awhile they find out. "
            "Watch for the character of the village idiot who clucks like a chicken, "
            "he certainly steals the show."
        ),
        "expected": "Sentiment réel : négatif (note basse). Le modèle hésite — confiance modérée.",
        "is_tricky": False,
    },
    "subtle_positive": {
        "label": "Éloge nuancé d'un film d'auteur",
        "category": "Cas nuancés (le modèle hésite légitimement)",
        "text": (
            "I admire Deepa Mehta and this movie is a masterpiece. I'd recommend to buy "
            "this movie on DVD because it's a movie you might want to watch more often "
            "than just once. And trust me, you'd still find little meaningful details "
            "after watching it several times. The characters - except for the grandmother "
            "and the maid - are not perfect."
        ),
        "expected": "Sentiment réel : positif. Texte plus subtil qu'une simple éloge directe.",
        "is_tricky": False,
    },
    "sarcasm": {
        "label": "Critique sarcastique",
        "category": "Cas difficiles (limites du modèle classique)",
        "text": (
            "There must be an error. This movie belongs with \"Plan 9\", and a lot others "
            "as a quite entertaining, silly diversion. You'll never accept you like it, "
            "yet you will watch it whenever it comes out on TV. It's as simple as that."
        ),
        "expected": (
            "Sentiment réel : **négatif**. La critique compare le film à Plan 9 from Outer Space "
            "(considéré comme l'un des pires films jamais réalisés). Le modèle interprète "
            "à tort les mots positifs (\"entertaining\", \"like it\") sans saisir l'ironie. "
            "C'est une limite connue des approches sac-de-mots."
        ),
        "is_tricky": True,
    },
    "ambivalent": {
        "label": "Critique ambivalente (\"guilty pleasure\")",
        "category": "Cas difficiles (limites du modèle classique)",
        "text": (
            "This film features two of my favorite guilty pleasures. Sure, the effects "
            "are laughable, the story confused, but just watching Hasselhoff in his "
            "Knight Rider days is always fun. I especially like the old hotel they used "
            "to shoot this in, it added to what little suspense was mustered. Give it a 3."
        ),
        "expected": (
            "Sentiment réel : **négatif** (note de 3/10 mentionnée à la fin). "
            "Le \"guilty pleasure\" exprime une appréciation ambivalente que les modèles "
            "classiques peinent à distinguer d'un sentiment positif."
        ),
        "is_tricky": True,
    },
}

CATEGORIES = [
    "Cas clairs (haute confiance attendue)",
    "Cas nuancés (le modèle hésite légitimement)",
    "Cas difficiles (limites du modèle classique)",
]


# ============================================================================
# UI principale
# ============================================================================
st.title("Analyse de sentiment de critiques de films")

st.markdown(
    "Cette application classe une critique de film comme **positive** ou "
    "**négative** à l'aide d'un modèle TF-IDF entraîné sur 22 000 critiques IMDB."
)

with st.container(border=True):
    st.markdown(
        "**À propos du modèle utilisé**\n\n"
        "Le modèle a été entraîné uniquement sur des critiques **en anglais** "
        "issues du corpus IMDB (films). Il atteint une exactitude de 92 % sur des "
        "critiques de films similaires, mais peut sous-performer sur :\n"
        "- des textes dans d'autres langues,\n"
        "- des domaines différents (produits, restaurants, services),\n"
        "- des tournures sarcastiques ou ironiques."
    )


# === Section : Exemples ===
st.markdown("### Tester avec un exemple")
st.caption(
    "Tous les exemples proviennent du test set IMDB — le modèle ne les a jamais "
    "vus pendant l'entraînement."
)

if "review_text" not in st.session_state:
    st.session_state["review_text"] = ""
if "current_example_key" not in st.session_state:
    st.session_state["current_example_key"] = None

for category in CATEGORIES:
    items = [(k, v) for k, v in EXAMPLES.items() if v["category"] == category]
    st.markdown(f"**{category}**")
    cols = st.columns(len(items))
    for col, (key, example) in zip(cols, items):
        with col:
            if st.button(example["label"], key=f"btn_{key}", use_container_width=True):
                st.session_state["review_text"] = example["text"]
                st.session_state["current_example_key"] = key


# === Section : Saisie ===
st.markdown("### Critique à analyser")

review = st.text_area(
    label="critique",
    value=st.session_state["review_text"],
    height=180,
    placeholder=(
        "Tapez une critique en anglais, ou choisissez un exemple ci-dessus."
    ),
    label_visibility="collapsed",
)

# Si l'utilisateur modifie le texte, on perd l'association à l'exemple
if review != st.session_state["review_text"]:
    st.session_state["current_example_key"] = None

analyse = st.button("Analyser le sentiment", type="primary", use_container_width=True)


# === Section : Résultat ===
if analyse:
    if not review.strip():
        st.warning("Veuillez entrer une critique ou choisir un exemple.")
    else:
        predict_fn = get_predict_fn()
        with st.spinner("Analyse en cours..."):
            result = predict_fn(review)

        st.markdown("### Résultat")

        label = result["label"]
        confidence = result["confidence"]
        proba_pos = result["probabilities"]["positif"]
        proba_neg = result["probabilities"]["négatif"]

        col1, col2 = st.columns([1, 1])
        with col1:
            st.metric(
                label="Sentiment prédit",
                value="Positif" if label == "positif" else "Négatif",
            )
        with col2:
            st.metric(label="Confiance", value=f"{confidence:.1%}")

        st.markdown("**Probabilités**")
        st.progress(proba_neg, text=f"Négatif — {proba_neg:.1%}")
        st.progress(proba_pos, text=f"Positif — {proba_pos:.1%}")

        # Note explicative pour les exemples piégeux
        example_key = st.session_state.get("current_example_key")
        if example_key and EXAMPLES[example_key]["is_tricky"]:
            st.warning(
                f"**Note sur cet exemple.** {EXAMPLES[example_key]['expected']}"
            )
        elif example_key:
            st.caption(EXAMPLES[example_key]["expected"])


# === Sections d'information rétractables ===
st.divider()

with st.expander("Comprendre le modèle"):
    st.markdown(
        "Le modèle utilisé est une **régression logistique** entraînée sur des "
        "vecteurs **TF-IDF** combinant unigrammes et bigrammes. Trois paradigmes "
        "ont été comparés sur ce projet :\n\n"
        "| Approche | Validation F1 | Notes |\n"
        "| :--- | :---: | :--- |\n"
        "| N-grammes simples (comptage) | 0.906 | Baseline solide |\n"
        "| **TF-IDF uni+bi sublinear** | **0.919** | Modèle utilisé ici |\n"
        "| BiLSTM + GloVe 300d | 0.914 | Statistiquement équivalent |\n"
        "| DistilBERT (en cours) | — | Saut technologique attendu |\n\n"
        "Le TF-IDF a été retenu pour cette démonstration en raison de son équilibre "
        "performance / taille (12 Mo contre 250 Mo pour DistilBERT)."
    )

with st.expander("Carte du modèle (model card)"):
    st.markdown(
        "**Usage prévu.** Classification binaire de critiques de films en anglais, "
        "à des fins éducatives ou de démonstration de pipeline NLP.\n\n"
        "**Données d'entraînement.** 22 000 critiques tirées du corpus public IMDB "
        "(Maas et al., 2011), équilibrées entre classes positive et négative.\n\n"
        "**Performance mesurée.** Exactitude 92.0 %, F1 macro 91.9 % sur un set de "
        "validation de 3 000 critiques. Performance sur le test set (25 000 critiques) "
        "à mesurer une fois le modèle final retenu.\n\n"
        "**Limitations connues.**\n"
        "- Modèle entraîné uniquement sur l'anglais. Sortie aléatoire sur d'autres "
        "langues.\n"
        "- Performance dégradée sur des textes hors du domaine cinéma "
        "(produits, services, etc.).\n"
        "- Sensible au sarcasme et à l'ironie : un texte avec des marqueurs "
        "lexicalement positifs peut être classé positif même si l'intention est "
        "ironique.\n"
        "- Le modèle ne distingue pas les textes très courts (< 20 mots) des "
        "textes longs ; les courts sont typiquement moins fiables.\n\n"
        "**Considérations éthiques.** Ce modèle ne doit pas être utilisé pour "
        "des décisions automatisées affectant des personnes (modération de contenu, "
        "scoring de feedback client en production) sans validation humaine et "
        "audit complémentaire."
    )

with st.expander("Détails techniques"):
    st.markdown(
        "**Préprocessing.** Mise en minuscules, retrait de la ponctuation, "
        "tokenisation par espaces.\n\n"
        "**Vectorisation.** TfidfVectorizer scikit-learn, ngram_range=(1,2), "
        "min_df=3, sublinear_tf=True. Vocabulaire de 242 975 features.\n\n"
        "**Classifieur.** Régression logistique PyTorch, optimiseur Adam, lr=1e-3, "
        "early stopping (patience 3, min_delta 1e-4), F1 macro comme critère de "
        "sélection.\n\n"
        "**Reproductibilité.** Toutes les expériences sont versionnées dans "
        "`outputs/experiments.json` du dépôt GitHub. Les modèles intermédiaires "
        "(bigramme baseline, TF-IDF, BiLSTM+GloVe) sont archivés localement."
    )


# === Footer ===
st.divider()

st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
        <p><strong>Sandra Desmair Fogang Lontouo</strong></p>
        <p>
            <a href='https://github.com/sandraFogang/nlp-sentiment-classification' target='_blank'>GitHub</a>
            &nbsp;·&nbsp;
            <a href='https://www.linkedin.com/in/sandrafogang' target='_blank'>LinkedIn</a>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)