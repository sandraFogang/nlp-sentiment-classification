"""Streamlit app — NLP Sentiment Analysis Dashboard."""
import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


st.set_page_config(
    page_title="NLP Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Couleurs partagées entre la jauge et les badges
COLOR_POSITIVE = "#2EC27E"
COLOR_NEGATIVE = "#E5484D"


# ============================================================================
# Compact dark dashboard CSS
# ============================================================================
st.markdown(
    f"""
    <style>
    .stApp, [data-testid="stAppViewContainer"] {{
        background-color: #0E1117 !important;
    }}
    .block-container {{
        padding-top: 0.4rem !important;
        padding-bottom: 0.4rem !important;
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
        max-width: 100% !important;
    }}
    .stApp, .stApp p, .stApp label {{
        color: #E0E0E0;
    }}

    /* Tighter vertical gaps between elements */
    [data-testid="stVerticalBlock"] > [data-testid="element-container"] {{
        margin-bottom: 0.2rem !important;
    }}
    /* Tighter horizontal gap between columns (was ~1rem default) */
    [data-testid="stHorizontalBlock"] {{
        gap: 0.5rem !important;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: #0A0D12 !important;
        min-width: 240px !important;
        max-width: 260px !important;
    }}
    section[data-testid="stSidebar"] > div {{
        padding-top: 0.6rem !important;
    }}

    /* Inputs */
    .stTextArea textarea {{
        background-color: #161B22 !important;
        color: #E0E0E0 !important;
        border: 1px solid #2D3138 !important;
        font-size: 0.82rem !important;
    }}
    .stSelectbox > div > div {{
        background-color: #161B22 !important;
        color: #E0E0E0 !important;
        border: 1px solid #2D3138 !important;
        font-size: 0.82rem !important;
    }}

    /* Buttons : enforce equal width and consistent appearance */
    .stButton button[kind="secondary"] {{
        background-color: #1B2028 !important;
        border: 1px solid #2D3138 !important;
        color: #E0E0E0 !important;
        font-size: 0.78rem !important;
        padding: 0.3rem 0.4rem !important;
        height: 34px !important;
        min-height: 34px !important;
        max-height: 34px !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }}
    .stButton button[kind="secondary"]:hover {{
        background-color: #252B36 !important;
        border-color: #3D424B !important;
    }}
    .stButton button[kind="primary"] {{
        background: linear-gradient(135deg, #A87FFF 0%, #6F4FBF 100%) !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
        padding: 0.4rem !important;
        font-size: 0.88rem !important;
        height: 38px !important;
    }}
    .stButton button[kind="primary"]:hover {{
        background: linear-gradient(135deg, #B894FF 0%, #7F5FCF 100%) !important;
    }}

    /* Hero */
    .hero-title {{
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin: 0;
    }}
    .hero-title h1 {{
        margin: 0 !important;
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        background: linear-gradient(90deg, #C9A0FF 0%, #A87FFF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}
    .hero-subtitle {{
        color: #888A92 !important;
        font-size: 0.74rem !important;
        margin: 0 0 0.4rem 0 !important;
    }}

    /* Streamlit container border = card */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background-color: #161B22 !important;
        border: 1px solid #2D3138 !important;
        border-radius: 10px !important;
        padding: 0.45rem 0.65rem !important;
    }}

    /* Best model summary */
    .best-model-box {{
        background: linear-gradient(135deg, #1B2330 0%, #161B22 100%);
        border: 1px solid #2D3138;
        border-radius: 10px;
        padding: 0.5rem 0.75rem;
        height: 100%;
    }}
    .best-model-label {{
        font-size: 0.62rem;
        color: #888A92;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }}
    .best-model-name {{
        font-size: 0.88rem;
        font-weight: 700;
        color: #E8E8EA;
        margin: 0.1rem 0 0;
    }}
    .best-model-f1 {{
        font-size: 1.1rem;
        font-weight: 700;
        color: {COLOR_POSITIVE};
        text-align: right;
        margin: 0;
    }}

    /* Card numbered headers */
    .card-num {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 18px;
        height: 18px;
        border-radius: 50%;
        background: linear-gradient(135deg, #A87FFF, #6F4FBF);
        color: white;
        font-size: 0.68rem;
        font-weight: 700;
        margin-right: 0.35rem;
        vertical-align: middle;
    }}
    .card-header-text {{
        font-size: 0.84rem;
        font-weight: 600;
        color: #E8E8EA;
    }}

    /* Sidebar */
    .sidebar-title {{
        font-size: 0.95rem;
        font-weight: 700;
        color: #C9A0FF;
        line-height: 1.2;
        margin-bottom: 0.1rem;
    }}
    .sidebar-subtitle {{
        color: #888A92 !important;
        font-size: 0.68rem;
        margin-bottom: 0.7rem;
    }}
    .sidebar-section-title {{
        font-size: 0.66rem !important;
        color: #888A92 !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
        margin: 0.65rem 0 0.25rem !important;
    }}

    /* Model info card */
    .model-info-card {{
        background-color: #161B22;
        border: 1px solid #2D3138;
        border-radius: 8px;
        padding: 0.5rem 0.65rem;
        margin: 0.25rem 0 0.5rem;
        font-size: 0.74rem;
    }}
    .model-info-title {{
        font-weight: 600;
        color: #E8E8EA;
        margin-bottom: 0.35rem;
        font-size: 0.78rem;
    }}
    .info-row {{
        display: flex;
        justify-content: space-between;
        padding: 0.1rem 0;
    }}
    .info-label {{ color: #888A92; }}
    .info-value {{ color: #E8E8EA; font-weight: 500; }}

    /* Quick links */
    .quick-links {{
        font-size: 0.74rem;
        line-height: 1.65;
    }}
    .quick-links a {{
        color: #C9A0FF;
        text-decoration: none;
    }}
    .quick-links a:hover {{
        color: #E5C7FF;
    }}

    /* Verdict */
    .verdict-icon {{
        font-size: 2.2rem;
        line-height: 1;
    }}
    .verdict-pos {{ color: {COLOR_POSITIVE}; }}
    .verdict-neg {{ color: {COLOR_NEGATIVE}; }}

    /* Probability bar */
    .proba-bar-track {{
        background-color: #2D3138;
        border-radius: 4px;
        height: 6px;
        overflow: hidden;
        margin-top: 4px;
    }}
    .proba-bar-fill-pos {{
        background: linear-gradient(90deg, {COLOR_POSITIVE}, #4DD095);
        height: 100%;
    }}

    /* Comparison table */
    .comp-row {{
        display: grid;
        grid-template-columns: 2.4fr 1fr 1fr 0.9fr 0.9fr;
        align-items: center;
        gap: 0.35rem;
        padding: 0.35rem 0.4rem;
        border-bottom: 1px solid #1F232A;
        font-size: 0.76rem;
    }}
    .comp-row.header {{
        color: #888A92;
        font-size: 0.64rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        font-weight: 600;
        border-bottom: 1px solid #2D3138;
    }}
    .winner-f1 {{
        color: #C9A0FF !important;
        font-weight: 700;
    }}
    /* Badges harmonisés avec la jauge */
    .pred-badge-pos {{
        background-color: rgba(46, 194, 126, 0.18);
        color: {COLOR_POSITIVE};
        padding: 0.12rem 0.45rem;
        border-radius: 10px;
        font-size: 0.7rem;
        font-weight: 600;
        display: inline-block;
    }}
    .pred-badge-neg {{
        background-color: rgba(229, 72, 77, 0.18);
        color: {COLOR_NEGATIVE};
        padding: 0.12rem 0.45rem;
        border-radius: 10px;
        font-size: 0.7rem;
        font-weight: 600;
        display: inline-block;
    }}
    .pred-badge-na {{ color: #555; font-size: 0.7rem; }}

    /* Tech panel */
    .tech-panel {{
        background-color: #161B22;
        border: 1px solid #2D3138;
        border-radius: 10px;
        padding: 0.55rem 0.75rem;
        margin-top: 0.35rem;
    }}
    .tech-panel-title {{
        font-size: 0.85rem;
        font-weight: 600;
        color: #E8E8EA;
        margin-bottom: 0.35rem;
    }}
    .tech-grid {{
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 0.7rem;
    }}
    .tech-item-title {{
        font-weight: 600;
        color: #E8E8EA;
        font-size: 0.74rem;
        margin-bottom: 0.18rem;
    }}
    .tech-item-desc {{
        color: #B0B3BB;
        font-size: 0.66rem;
        line-height: 1.35;
    }}

    /* Footer */
    .footer-bar {{
        text-align: center;
        margin-top: 0.35rem;
        padding: 0.3rem 0;
        color: #888A92;
        font-size: 0.7rem;
        border-top: 1px solid #2D3138;
    }}
    .footer-bar a {{
        color: #C9A0FF;
        text-decoration: none;
        margin: 0 0.4rem;
    }}

    /* Hide chrome */
    #MainMenu, footer, header[data-testid="stHeader"] {{ display: none !important; }}
    [data-testid="stToolbar"] {{ display: none !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# Models
# ============================================================================
MODELS_FOR_SELECTOR = {
    "tfidf": {
        "display_name": "TF-IDF + Logistic Regression",
        "type": "Classical ML",
        "size_mb": 12,
        "latency": "~10 ms",
        "f1_val": "91.9 %",
        "params": "243 k",
        "available_on_cloud": True,
        "supports_interpretability": True,
    },
    "bert_finetuned": {
        "display_name": "DistilBERT (fine-tuned)",
        "type": "Transformer",
        "size_mb": 250,
        "latency": "~500 ms",
        "f1_val": "93.2 %",
        "params": "66.4 M",
        "available_on_cloud": False,
        "supports_interpretability": False,
    },
}

ALL_MODELS_COMPARISON = [
    {"name": "TF-IDF + Logistic Regression", "f1": "91.9 %", "latency": "~10 ms", "is_winner": False, "key": "tfidf"},
    {"name": "BiLSTM + GloVe 300d", "f1": "91.4 %", "latency": "~150 ms", "is_winner": False, "key": "bilstm"},
    {"name": "DistilBERT (frozen)", "f1": "85.4 %", "latency": "~500 ms", "is_winner": False, "key": "bert_frozen"},
    {"name": "DistilBERT (fine-tuned)", "f1": "93.2 %", "latency": "~500 ms", "is_winner": True, "key": "bert_finetuned"},
]


# ============================================================================
# Loaders
# ============================================================================
@st.cache_resource(show_spinner="Loading TF-IDF model...")
def _ensure_model_available():
    from nlp_sentiment.config import MODEL_PATH, VOCAB_PATH
    if not (MODEL_PATH.exists() and VOCAB_PATH.exists()):
        from train_champion_tfidf import main as train_champion
        train_champion()


@st.cache_resource
def get_predict_fn():
    _ensure_model_available()
    from nlp_sentiment.predict import predict
    return predict


@st.cache_resource
def get_interpretability_fn():
    _ensure_model_available()
    from nlp_sentiment.interpretability import get_word_contributions
    return get_word_contributions


# ============================================================================
# Examples
# ============================================================================
EXAMPLES = {
    "Positive": "The acting was phenomenal, the story kept me engaged from start to finish, and the cinematography was breathtaking. Absolutely one of the best movies I have ever seen!",
    "Negative": "What a complete waste of time. The plot was incoherent, the acting was wooden, and the pacing was excruciatingly slow. I checked my watch every few minutes hoping it would end.",
    "Sarcastic": "There must be an error. This movie belongs with \"Plan 9\", and a lot others as a quite entertaining, silly diversion. You'll never accept you like it, yet you will watch it whenever it comes out on TV.",
    "Neutral": "The film has its moments. Some scenes are visually impressive and the lead performance shows real talent. However, the screenplay feels underdeveloped and the second act drags considerably.",
}


# ============================================================================
# Plotly gauge — compact, no yellow band
# ============================================================================
def render_gauge(confidence: float, is_positive: bool) -> go.Figure:
    color = COLOR_POSITIVE if is_positive else COLOR_NEGATIVE

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={"x": [0, 1], "y": [0, 1]},
        number={
            "suffix": "%",
            "font": {"size": 18, "color": "#E8E8EA", "family": "Arial"},
            "valueformat": ".1f",
        },
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 0,
                "tickvals": [0, 50, 100],
                "tickfont": {"color": "#888A92", "size": 7},
            },
            "bar": {"color": color, "thickness": 0.7},
            "bgcolor": "#0E1117",
            "borderwidth": 0,
            "steps": [],
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 10, "r": 10, "t": 5, "b": 5},
        height=105,
    )
    return fig


# ============================================================================
# Session state
# ============================================================================
# IMPORTANT : la cle "textarea_widget" est utilisee directement par le widget.
# Modifier session_state["textarea_widget"] AVANT le prochain render
# fait que le widget se met a jour visuellement.
if "textarea_widget" not in st.session_state:
    st.session_state["textarea_widget"] = ""
if "selected_model" not in st.session_state:
    st.session_state["selected_model"] = "tfidf"
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "last_error" not in st.session_state:
    st.session_state["last_error"] = None


def _run_prediction(text: str):
    """Helper : exécute la prédiction et stocke le résultat."""
    selected = MODELS_FOR_SELECTOR[st.session_state["selected_model"]]
    if not selected["available_on_cloud"]:
        st.session_state["last_result"] = None
        st.session_state["last_error"] = (
            f"{selected['display_name']} is not active on Streamlit Cloud. "
            "Switch to TF-IDF or run locally."
        )
        return
    predict_fn = get_predict_fn()
    st.session_state["last_result"] = predict_fn(text)
    st.session_state["last_error"] = None


def load_example_and_analyze(label: str):
    """Charge l'exemple dans la textarea ET déclenche l'analyse."""
    text = EXAMPLES[label]
    # Cle directe du widget = mise a jour visuelle au prochain render
    st.session_state["textarea_widget"] = text
    _run_prediction(text)


def trigger_analyze():
    """Analyse à partir du texte actuel de la textarea."""
    text = st.session_state.get("textarea_widget", "").strip()
    if not text:
        st.session_state["last_result"] = None
        st.session_state["last_error"] = "Please enter a review or pick an example."
        return
    _run_prediction(text)


# ============================================================================
# Sidebar
# ============================================================================
with st.sidebar:
    st.markdown(
        """
        <div class='sidebar-title'>🧠 NLP Sentiment<br/>Analyzer</div>
        <div class='sidebar-subtitle'>Multi-paradigm Analysis</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='sidebar-section-title'>MODEL SELECTION</div>", unsafe_allow_html=True)

    options = {
        key: info["display_name"] + (" 🔒" if not info["available_on_cloud"] else "")
        for key, info in MODELS_FOR_SELECTOR.items()
    }
    selected_key = st.selectbox(
        label="Model",
        options=list(options.keys()),
        format_func=lambda k: options[k],
        index=list(options.keys()).index(st.session_state["selected_model"]),
        label_visibility="collapsed",
    )
    st.session_state["selected_model"] = selected_key
    selected_info = MODELS_FOR_SELECTOR[selected_key]

    if not selected_info["available_on_cloud"]:
        st.warning(
            f"**Local-only.** {selected_info['display_name']} ({selected_info['size_mb']} MB) "
            "is not deployed. Run locally to use it.",
            icon="🔒",
        )

    st.markdown(
        f"""
        <div class='model-info-card'>
            <div class='model-info-title'>📊 Model Info</div>
            <div class='info-row'><span class='info-label'>Type</span><span class='info-value'>{selected_info['type']}</span></div>
            <div class='info-row'><span class='info-label'>Size</span><span class='info-value'>{selected_info['size_mb']} MB</span></div>
            <div class='info-row'><span class='info-label'>Latency</span><span class='info-value'>{selected_info['latency']}</span></div>
            <div class='info-row'><span class='info-label'>F1 (Val)</span><span class='info-value'>{selected_info['f1_val']}</span></div>
            <div class='info-row'><span class='info-label'>Params</span><span class='info-value'>{selected_info['params']}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if selected_key != "tfidf":
        tfidf = MODELS_FOR_SELECTOR["tfidf"]
        st.markdown(
            f"""
            <div class='model-info-card'>
                <div class='model-info-title'>🪶 Currently Deployed</div>
                <div class='info-row'><span class='info-label'>Model</span><span class='info-value'>TF-IDF</span></div>
                <div class='info-row'><span class='info-label'>Size</span><span class='info-value'>{tfidf['size_mb']} MB</span></div>
                <div class='info-row'><span class='info-label'>F1</span><span class='info-value'>{tfidf['f1_val']}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div class='sidebar-section-title'>QUICK LINKS</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='quick-links'>
        🔗 <a href='https://github.com/sandraFogang/nlp-sentiment-classification' target='_blank'>View on GitHub</a><br/>
        📖 <a href='https://github.com/sandraFogang/nlp-sentiment-classification#readme' target='_blank'>Documentation</a><br/>
        📊 <a href='https://github.com/sandraFogang/nlp-sentiment-classification/blob/main/outputs/experiments.json' target='_blank'>Experiments</a><br/>
        💼 <a href='https://www.linkedin.com/in/sandrafogang' target='_blank'>LinkedIn</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# Hero header
# ============================================================================
hero_col, summary_col = st.columns([3, 1.5])
with hero_col:
    st.markdown(
        """
        <div class='hero-title'>
            <span style='font-size: 1.6rem;'>🎬</span>
            <h1>NLP Sentiment Analyzer</h1>
        </div>
        <p class='hero-subtitle'>Multi-paradigm sentiment analysis on IMDB movie reviews</p>
        """,
        unsafe_allow_html=True,
    )

with summary_col:
    st.markdown(
        """
        <div class='best-model-box'>
            <div style='display: flex; justify-content: space-between; align-items: flex-start;'>
                <div>
                    <div class='best-model-label'>Best Model (Validation Set)</div>
                    <div class='best-model-name'>DistilBERT (fine-tuned)</div>
                </div>
                <div style='text-align: right;'>
                    <div class='best-model-label'>F1 Score</div>
                    <div class='best-model-f1'>93.2%</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


result = st.session_state["last_result"]
error = st.session_state["last_error"]


# ============================================================================
# Row 1 — Input + Prediction
# ============================================================================
input_col, pred_col = st.columns([1, 1])

with input_col:
    with st.container(border=True):
        st.markdown(
            "<span class='card-num'>1</span><span class='card-header-text'>Enter a movie review</span>",
            unsafe_allow_html=True,
        )

        # IMPORTANT : pas de value=, uniquement key=
        # session_state["textarea_widget"] est la source de verite
        st.text_area(
            label="review",
            height=85,
            placeholder="Type or paste an English movie review here, or pick an example below.",
            label_visibility="collapsed",
            key="textarea_widget",
            max_chars=2000,
        )
        char_count = len(st.session_state["textarea_widget"])
        st.markdown(
            f"<div style='color: #888A92; font-size: 0.7rem; margin: 0.1rem 0;'>"
            f"{char_count} / 2000 characters · Try examples:</div>",
            unsafe_allow_html=True,
        )

        # 4 boutons en colonnes egales (gap supprime via CSS)
        ex1, ex2, ex3, ex4 = st.columns(4, gap="small")
        with ex1:
            st.button("Positive 😊", key="ex_pos", use_container_width=True,
                      on_click=load_example_and_analyze, args=("Positive",))
        with ex2:
            st.button("Negative 😞", key="ex_neg", use_container_width=True,
                      on_click=load_example_and_analyze, args=("Negative",))
        with ex3:
            st.button("Sarcastic 😏", key="ex_sar", use_container_width=True,
                      on_click=load_example_and_analyze, args=("Sarcastic",))
        with ex4:
            st.button("Neutral 😐", key="ex_neu", use_container_width=True,
                      on_click=load_example_and_analyze, args=("Neutral",))

        st.button(
            "🚀 Analyze Sentiment",
            type="primary",
            use_container_width=True,
            on_click=trigger_analyze,
            key="analyze_btn",
        )


with pred_col:
    with st.container(border=True):
        st.markdown(
            "<span class='card-num'>2</span><span class='card-header-text'>Prediction & Confidence</span>",
            unsafe_allow_html=True,
        )

        if result:
            label = result["label"]
            confidence = result["confidence"]
            proba_pos = result["probabilities"]["positif"]
            proba_neg = result["probabilities"]["négatif"]
            is_positive = label == "positif"
            verdict_class = "verdict-pos" if is_positive else "verdict-neg"
            verdict_text = "POSITIVE" if is_positive else "NEGATIVE"
            verdict_icon = "🙂" if is_positive else "🙁"

            v_col, g_col = st.columns([1, 1])
            with v_col:
                st.markdown(
                    f"<div style='text-align: center; padding-top: 0.3rem;'>"
                    f"<div class='verdict-icon {verdict_class}'>{verdict_icon}</div>"
                    f"<div class='{verdict_class}' style='font-size: 1.05rem; font-weight: 700; margin-top: 0.15rem;'>{verdict_text}</div>"
                    f"<div style='color: #B0B3BB; font-size: 0.66rem; margin-top: 0.2rem;'>"
                    f"{'Positive sentiment.' if is_positive else 'Negative sentiment.'}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with g_col:
                st.plotly_chart(render_gauge(confidence, is_positive),
                                use_container_width=True, config={"displayModeBar": False})

            st.markdown(
                f"<div style='display: flex; justify-content: space-between; padding: 0 0.4rem; margin-top: 0.2rem;'>"
                f"<div><div style='color: #B0B3BB; font-size: 0.66rem;'>Positive</div>"
                f"<div style='color: {COLOR_POSITIVE}; font-size: 0.95rem; font-weight: 700;'>{proba_pos:.1%}</div></div>"
                f"<div style='text-align: right;'><div style='color: #B0B3BB; font-size: 0.66rem;'>Negative</div>"
                f"<div style='color: {COLOR_NEGATIVE}; font-size: 0.95rem; font-weight: 700;'>{proba_neg:.1%}</div></div>"
                f"</div>"
                f"<div style='padding: 0 0.4rem;'>"
                f"<div class='proba-bar-track'><div class='proba-bar-fill-pos' style='width: {proba_pos * 100}%;'></div></div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        elif error:
            st.markdown(
                f"<div style='padding: 0.6rem; color: {COLOR_NEGATIVE}; font-size: 0.8rem;'>⚠️ {error}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='text-align: center; padding: 0.9rem 0.5rem; color: #888A92;'>"
                "<div style='font-size: 1.6rem;'>📊</div>"
                "<p style='margin-top: 0.3rem; font-size: 0.74rem;'>Awaiting input… "
                "Pick an example or type a review.</p>"
                "</div>",
                unsafe_allow_html=True,
            )


# ============================================================================
# Row 2 — Interpretability + Comparison
# ============================================================================
interp_col, comp_col = st.columns([1, 1])

with interp_col:
    with st.container(border=True):
        st.markdown(
            "<span class='card-num'>3</span><span class='card-header-text'>Why this prediction?</span> "
            "<span style='color: #888A92; font-size: 0.72rem;'>(Interpretability)</span>",
            unsafe_allow_html=True,
        )

        if result and selected_info["supports_interpretability"]:
            interp_fn = get_interpretability_fn()
            try:
                pos_words, neg_words = interp_fn(st.session_state["textarea_widget"], top_k=5)
                tab_pos, tab_neg = st.tabs(["Positive", "Negative"])

                def _render_words(words, color):
                    if not words:
                        st.caption("No contribution detected.")
                        return
                    max_c = max(w["contribution"] for w in words)
                    for w in words:
                        bar = (w["contribution"] / max_c) * 100
                        st.markdown(
                            f"<div style='display: flex; align-items: center; gap: 0.35rem; margin: 0.18rem 0;'>"
                            f"<div style='width: 85px; color: #E0E0E0; font-weight: 600; font-size: 0.74rem;'>{w['word']}</div>"
                            f"<div style='flex: 1; background-color: #2D3138; border-radius: 4px; height: 4px;'>"
                            f"<div style='width: {bar}%; background: {color}; height: 100%; border-radius: 4px;'></div></div>"
                            f"<div style='width: 35px; text-align: right; color: {color}; font-weight: 600; font-size: 0.7rem;'>{w['contribution']:.2f}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                with tab_pos:
                    _render_words(pos_words, COLOR_POSITIVE)
                with tab_neg:
                    _render_words(neg_words, COLOR_NEGATIVE)
            except Exception as e:
                st.error(f"Interpretability error: {e}")
        elif result and not selected_info["supports_interpretability"]:
            st.info(
                f"Interpretability not available for {selected_info['display_name']}. "
                "BERT requires LIME/SHAP.",
                icon="ℹ️",
            )
        else:
            st.markdown(
                "<div style='text-align: center; padding: 0.9rem 0.5rem; color: #888A92;'>"
                "<div style='font-size: 1.4rem;'>🔍</div>"
                "<p style='margin-top: 0.3rem; font-size: 0.74rem;'>Awaiting input… "
                "Run an analysis to see word contributions.</p>"
                "</div>",
                unsafe_allow_html=True,
            )


with comp_col:
    with st.container(border=True):
        st.markdown(
            "<span class='card-num'>4</span><span class='card-header-text'>Compare with other models</span> "
            "<span style='color: #888A92; font-size: 0.72rem;'>(Validation Set)</span>",
            unsafe_allow_html=True,
        )

        live_predictions = {}
        if result and selected_info["available_on_cloud"]:
            live_predictions[selected_key] = {
                "label": "Positive" if result["label"] == "positif" else "Negative",
                "confidence": f"{result['confidence']:.1%}",
            }

        # HTML construit en une seule chaine pour eviter
        # toute interpretation markdown indesirable.
        html_parts = ["<div class='comp-row header'><div>Model</div><div>Prediction</div><div>Confidence</div><div>F1 (Val)</div><div>Latency</div></div>"]

        for model in ALL_MODELS_COMPARISON:
            if model["key"] in live_predictions:
                pred = live_predictions[model["key"]]
                badge = f"<span class='pred-badge-pos'>Positive</span>" if pred["label"] == "Positive" else f"<span class='pred-badge-neg'>Negative</span>"
                conf = pred["confidence"]
            else:
                badge = "<span class='pred-badge-na'>—</span>"
                conf = "<span class='pred-badge-na'>—</span>"

            f1_class = "winner-f1" if model["is_winner"] else ""
            winner = " 🏆" if model["is_winner"] else ""
            html_parts.append(
                f"<div class='comp-row'><div style='color:#E0E0E0;'>{model['name']}{winner}</div>"
                f"<div>{badge}</div><div>{conf}</div>"
                f"<div class='{f1_class}'>{model['f1']}</div>"
                f"<div style='color:#B0B3BB;'>{model['latency']}</div></div>"
            )

        st.markdown("".join(html_parts), unsafe_allow_html=True)

        st.markdown(
            "<div style='color: #888A92; font-size: 0.62rem; margin-top: 0.3rem;'>"
            "ⓘ F1 macro on validation (3 000 reviews). Live prediction shown only for the selected model."
            "</div>",
            unsafe_allow_html=True,
        )


# ============================================================================
# Technical Details (always visible)
# ============================================================================
st.markdown(
    """
    <div class='tech-panel'>
        <div class='tech-panel-title'>📋 Technical Details &amp; Model Card</div>
        <div class='tech-grid'>
            <div><div class='tech-item-title'>📂 Dataset</div><div class='tech-item-desc'>IMDB Reviews (Maas et al., 2011) · 22k / 3k / 25k splits</div></div>
            <div><div class='tech-item-title'>⚙️ Preprocessing</div><div class='tech-item-desc'>Lowercase, punctuation removal, tokenization. WordPiece for BERT.</div></div>
            <div><div class='tech-item-title'>🧬 Models Evaluated</div><div class='tech-item-desc'>4 paradigms: n-grams, TF-IDF, BiLSTM (GloVe), DistilBERT.</div></div>
            <div><div class='tech-item-title'>📊 Metrics</div><div class='tech-item-desc'>Accuracy, Precision, Recall, F1 macro, Latency, Size.</div></div>
            <div><div class='tech-item-title'>⚠️ Limitations</div><div class='tech-item-desc'>Sarcasm, irony, short texts, English-only, dataset bias.</div></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# Footer
# ============================================================================
st.markdown(
    """
    <div class='footer-bar'>
        Designed by <strong style='color: #E8E8EA;'>Sandra Fogang</strong> · Turning text into insights ·
        <a href='https://www.linkedin.com/in/sandrafogang' target='_blank'>LinkedIn</a> ·
        <a href='https://github.com/sandraFogang/nlp-sentiment-classification' target='_blank'>GitHub</a>
    </div>
    """,
    unsafe_allow_html=True,
)