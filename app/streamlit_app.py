"""Sentiment analysis web app — IMDB movie reviews."""
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


st.set_page_config(
    page_title="Sentiment Analysis — Movie Reviews",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# Custom CSS for a sober, professional look
# ============================================================================
st.markdown(
    """
    <style>
    /* Reduce default top padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }

    /* Hero header */
    .hero-header {
        background: linear-gradient(135deg, #f7f7f5 0%, #ececea 100%);
        border-radius: 12px;
        padding: 1.5rem 1.75rem;
        margin-bottom: 1.5rem;
        border: 1px solid #e3e3df;
    }
    .hero-header h1 {
        margin: 0;
        font-size: 1.75rem;
        font-weight: 600;
        color: #1a1a1a;
        letter-spacing: -0.01em;
    }
    .hero-header p {
        margin: 0.5rem 0 0;
        color: #5a5a55;
        font-size: 0.95rem;
    }
    .hero-stats {
        display: flex;
        gap: 1.5rem;
        margin-top: 0.75rem;
        font-size: 0.85rem;
        color: #6a6a65;
    }
    .hero-stats span strong {
        color: #1a1a1a;
    }

    /* Sidebar headers */
    section[data-testid="stSidebar"] h3 {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #6a6a65;
        font-weight: 600;
        margin-top: 1.2rem;
        margin-bottom: 0.5rem;
    }

    /* Sidebar example category labels */
    .example-category {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #8a8a85;
        font-weight: 600;
        margin: 0.85rem 0 0.4rem;
        padding-bottom: 0.25rem;
        border-bottom: 1px solid #e8e8e4;
    }

    /* Make sidebar buttons feel less heavy */
    section[data-testid="stSidebar"] button[kind="secondary"] {
        background: transparent;
        border: 1px solid #e3e3df;
        text-align: left;
        font-size: 0.85rem;
        padding: 0.45rem 0.7rem;
    }
    section[data-testid="stSidebar"] button[kind="secondary"]:hover {
        background: #f5f5f2;
        border-color: #c5c5bf;
    }

    /* Result cards */
    .result-section {
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# Model loading (with intelligent fallback)
# ============================================================================
@st.cache_resource(show_spinner="Loading model...")
def _ensure_model_available() -> None:
    from nlp_sentiment.config import MODEL_PATH, VOCAB_PATH

    if not (MODEL_PATH.exists() and VOCAB_PATH.exists()):
        st.info("First-time setup: training the model (about 3 minutes).")
        from train_champion_tfidf import main as train_champion
        train_champion()


@st.cache_resource
def get_predict_fn():
    _ensure_model_available()
    from nlp_sentiment.predict import predict
    return predict


# ============================================================================
# Examples from IMDB test set (never seen during training)
# ============================================================================
EXAMPLES = {
    "clear_positive": {
        "label": "Positive review (clear case)",
        "category": "Clear cases",
        "text": (
            "Previous reviewer Claudio Carvalho gave a much better recap of the film's "
            "plot details than I could. What I recall mostly is that it was just so "
            "beautiful, in every sense - emotionally, visually, editorially - just gorgeous. "
            "If you like movies that are wonderful to look at, and also have emotional "
            "content to which that beauty is relevant, I think you will be glad to have "
            "seen this extraordinary and unusual work of art."
        ),
        "expected": "Actual sentiment: positive. The model should be highly confident.",
        "is_tricky": False,
    },
    "clear_negative": {
        "label": "Negative review (clear case)",
        "category": "Clear cases",
        "text": (
            "Technically I'am a Van Damme Fan, or I was. This movie is so bad that I hated "
            "myself for wasting those 90 minutes. Do not let the name Isaac Florentine "
            "(Undisputed II) fool you, I had big hopes for this one, depending on what I "
            "saw in (Undisputed II), man.. was I wrong ??! All action fans wanted a better "
            "Van Damme movie. The story is weak, the fights are mediocre and the directing "
            "is below average. A complete disappointment."
        ),
        "expected": "Actual sentiment: negative. The model should be highly confident.",
        "is_tricky": False,
    },
    "moderate_negative": {
        "label": "Short ambiguous review",
        "category": "Nuanced cases",
        "text": (
            "An obscure horror show filmed in the Everglades. Two couples stay overnight "
            "in a cabin after being made a little uneasy by the unfriendliness of the "
            "locals. Who, or what, are the Blood Stalkers? After awhile they find out. "
            "Watch for the character of the village idiot who clucks like a chicken, "
            "he certainly steals the show."
        ),
        "expected": "Actual sentiment: negative (low rating). The model hesitates with moderate confidence.",
        "is_tricky": False,
    },
    "subtle_positive": {
        "label": "Nuanced praise",
        "category": "Nuanced cases",
        "text": (
            "I admire Deepa Mehta and this movie is a masterpiece. I'd recommend to buy "
            "this movie on DVD because it's a movie you might want to watch more often "
            "than just once. And trust me, you'd still find little meaningful details "
            "after watching it several times. The characters - except for the grandmother "
            "and the maid - are not perfect."
        ),
        "expected": "Actual sentiment: positive. More subtle than direct praise.",
        "is_tricky": False,
    },
    "sarcasm": {
        "label": "Sarcastic review",
        "category": "Hard cases (model limits)",
        "text": (
            "There must be an error. This movie belongs with \"Plan 9\", and a lot others "
            "as a quite entertaining, silly diversion. You'll never accept you like it, "
            "yet you will watch it whenever it comes out on TV. It's as simple as that."
        ),
        "expected": (
            "Actual sentiment: **negative**. The reviewer compares the film to *Plan 9 from Outer Space* "
            "(considered one of the worst movies ever made). The model misreads positive words "
            "(\"entertaining\", \"like it\") without grasping the irony. This is a known limitation "
            "of bag-of-words approaches."
        ),
        "is_tricky": True,
    },
    "ambivalent": {
        "label": "Ambivalent review (\"guilty pleasure\")",
        "category": "Hard cases (model limits)",
        "text": (
            "This film features two of my favorite guilty pleasures. Sure, the effects "
            "are laughable, the story confused, but just watching Hasselhoff in his "
            "Knight Rider days is always fun. I especially like the old hotel they used "
            "to shoot this in, it added to what little suspense was mustered. Give it a 3."
        ),
        "expected": (
            "Actual sentiment: **negative** (rating of 3/10 stated at the end). "
            "The \"guilty pleasure\" framing expresses ambivalent appreciation that classical "
            "models struggle to distinguish from positive sentiment."
        ),
        "is_tricky": True,
    },
}

CATEGORIES = ["Clear cases", "Nuanced cases", "Hard cases (model limits)"]


# ============================================================================
# Session state initialization
# ============================================================================
if "review_text" not in st.session_state:
    st.session_state["review_text"] = ""
if "current_example_key" not in st.session_state:
    st.session_state["current_example_key"] = None


# ============================================================================
# Sidebar — Examples and documentation
# ============================================================================
with st.sidebar:
    st.markdown("### Try an example")
    st.caption("All examples come from the IMDB test set. The model never saw them during training.")

    for category in CATEGORIES:
        st.markdown(f"<div class='example-category'>{category}</div>", unsafe_allow_html=True)
        items = [(k, v) for k, v in EXAMPLES.items() if v["category"] == category]
        for key, example in items:
            if st.button(example["label"], key=f"btn_{key}", use_container_width=True):
                st.session_state["review_text"] = example["text"]
                st.session_state["current_example_key"] = key

    st.markdown("---")
    st.markdown("### Documentation")

    with st.expander("About the model"):
        st.markdown(
            "The model is a **logistic regression** trained on **TF-IDF vectors** "
            "combining unigrams and bigrams. Three paradigms were compared on this project:\n\n"
            "| Approach | Validation F1 | Notes |\n"
            "| :--- | :---: | :--- |\n"
            "| Simple n-grams (counts) | 0.906 | Strong baseline |\n"
            "| **TF-IDF uni+bi sublinear** | **0.919** | Used here |\n"
            "| BiLSTM + GloVe 300d | 0.914 | Statistically equivalent |\n"
            "| DistilBERT (in progress) | — | Expected next step |\n\n"
            "TF-IDF was chosen for this demo for its balance of performance and size "
            "(12 MB vs 250 MB for DistilBERT)."
        )

    with st.expander("Model card"):
        st.markdown(
            "**Intended use.** Binary sentiment classification of English movie reviews, "
            "for educational or NLP pipeline demonstration purposes.\n\n"
            "**Training data.** 22,000 reviews from the public IMDB corpus "
            "(Maas et al., 2011), balanced between positive and negative classes.\n\n"
            "**Measured performance.** Accuracy 92.0%, macro F1 91.9% on a validation "
            "set of 3,000 reviews. Test set performance (25,000 reviews) to be measured "
            "once the final model is selected.\n\n"
            "**Known limitations.**\n"
            "- Trained on English only. Output is unreliable on other languages.\n"
            "- Degraded performance on out-of-domain texts (products, services, etc.).\n"
            "- Sensitive to sarcasm and irony: a text with lexically positive markers "
            "may be classified as positive even if the intent is ironic.\n"
            "- Does not differentiate very short texts (< 20 words) from long ones; "
            "short texts are typically less reliable.\n\n"
            "**Ethical considerations.** This model should not be used for automated "
            "decisions affecting individuals (content moderation, customer feedback "
            "scoring in production) without human validation and additional auditing."
        )

    with st.expander("Technical details"):
        st.markdown(
            "**Preprocessing.** Lowercase conversion, punctuation removal, "
            "whitespace tokenization.\n\n"
            "**Vectorization.** scikit-learn's TfidfVectorizer, ngram_range=(1,2), "
            "min_df=3, sublinear_tf=True. Vocabulary of 242,975 features.\n\n"
            "**Classifier.** PyTorch logistic regression, Adam optimizer, lr=1e-3, "
            "early stopping (patience 3, min_delta 1e-4), macro F1 as selection criterion.\n\n"
            "**Reproducibility.** All experiments are versioned in "
            "`outputs/experiments.json` of the GitHub repository. Intermediate champions "
            "(bigram baseline, TF-IDF, BiLSTM+GloVe) are archived locally."
        )

    st.markdown("---")
    st.markdown(
        "<p style='font-size: 0.8rem; color: #6a6a65; margin-top: 1rem;'>"
        "<strong>Sandra Desmair Fogang Lontouo</strong><br>"
        "<a href='https://github.com/sandraFogang/nlp-sentiment-classification' "
        "target='_blank' style='color: #4a4a45;'>GitHub</a> · "
        "<a href='https://www.linkedin.com/in/sandrafogang' "
        "target='_blank' style='color: #4a4a45;'>LinkedIn</a>"
        "</p>",
        unsafe_allow_html=True,
    )


# ============================================================================
# Main area — Hero header + input + results
# ============================================================================
st.markdown(
    """
    <div class='hero-header'>
        <h1>Sentiment Analysis of Movie Reviews</h1>
        <p>Classify a movie review as <strong>positive</strong> or <strong>negative</strong> using a TF-IDF model trained on IMDB.</p>
        <div class='hero-stats'>
            <span>Training data: <strong>22,000 reviews</strong></span>
            <span>Validation F1: <strong>91.9%</strong></span>
            <span>Language: <strong>English</strong></span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Limitations notice (compact)
st.info(
    "**Note.** This model was trained exclusively on **English** movie reviews. "
    "It may underperform on other languages, out-of-domain texts (products, services), "
    "or texts using sarcasm and irony.",
    icon="ℹ️",
)

# === Input area ===
review = st.text_area(
    label="Review to analyze",
    value=st.session_state["review_text"],
    height=170,
    placeholder=(
        "Paste an English movie review here, or pick an example from the sidebar."
    ),
    label_visibility="visible",
)

# Detect manual edit (drop association with the loaded example)
if review != st.session_state["review_text"]:
    st.session_state["current_example_key"] = None
    st.session_state["review_text"] = review

analyze = st.button(
    "Analyze sentiment",
    type="primary",
    use_container_width=True,
)


# === Results ===
if analyze:
    if not review.strip():
        st.warning("Please enter a review or pick an example from the sidebar.")
    else:
        predict_fn = get_predict_fn()
        with st.spinner("Analyzing..."):
            result = predict_fn(review)

        st.markdown("<div class='result-section'></div>", unsafe_allow_html=True)
        st.markdown("### Result")

        label = result["label"]
        confidence = result["confidence"]
        proba_pos = result["probabilities"]["positif"]
        proba_neg = result["probabilities"]["négatif"]

        sentiment_label = "Positive" if label == "positif" else "Negative"

        col1, col2 = st.columns([1, 1])
        with col1:
            st.metric(label="Predicted sentiment", value=sentiment_label)
        with col2:
            st.metric(label="Confidence", value=f"{confidence:.1%}")

        st.markdown("**Class probabilities**")
        st.progress(proba_neg, text=f"Negative — {proba_neg:.1%}")
        st.progress(proba_pos, text=f"Positive — {proba_pos:.1%}")

        # Explanatory note for tricky examples
        example_key = st.session_state.get("current_example_key")
        if example_key and EXAMPLES[example_key]["is_tricky"]:
            st.warning(
                f"**Note on this example.** {EXAMPLES[example_key]['expected']}",
                icon="⚠️",
            )
        elif example_key:
            st.caption(EXAMPLES[example_key]["expected"])