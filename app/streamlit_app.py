"""NLP Sentiment Analysis Dashboard.

Multi-paradigm sentiment classification with live inference for
TF-IDF, BiLSTM + GloVe, and DistilBERT (fine-tuned). Models are
hosted on the Hugging Face Hub and loaded lazily on selection.
"""
from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


st.set_page_config(
    page_title="NLP Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


COLOR_POS = "#2EC27E"
COLOR_NEG = "#E5484D"
COLOR_PURPLE = "#A87FFF"


st.markdown(
    f"""
    <style>
    .stApp, [data-testid="stAppViewContainer"] {{ background-color: #0E1117 !important; }}
    .block-container {{
        padding: 0.3rem 0.6rem !important;
        max-width: 100% !important;
    }}
    .stApp, .stApp p, .stApp label {{ color: #E0E0E0; }}

    [data-testid="stVerticalBlock"] > [data-testid="element-container"] {{ margin-bottom: 0.15rem !important; }}
    [data-testid="stHorizontalBlock"] {{ gap: 0.5rem !important; }}

    section[data-testid="stSidebar"] {{
        background-color: #0A0D12 !important;
        min-width: 230px !important;
        max-width: 250px !important;
    }}
    section[data-testid="stSidebar"] > div {{ padding-top: 0.5rem !important; }}

    .stTextArea textarea {{
        background-color: #161B22 !important;
        color: #E0E0E0 !important;
        border: 1px solid #2D3138 !important;
        font-size: 0.8rem !important;
    }}
    .stSelectbox > div > div {{
        background-color: #161B22 !important;
        color: #E0E0E0 !important;
        border: 1px solid #2D3138 !important;
        font-size: 0.8rem !important;
    }}

    .stButton button[kind="secondary"] {{
        background-color: #1B2028 !important;
        border: 1px solid #2D3138 !important;
        color: #E0E0E0 !important;
        font-size: 0.76rem !important;
        padding: 0.25rem 0.35rem !important;
        height: 32px !important;
        min-height: 32px !important;
        max-height: 32px !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }}
    .stButton button[kind="secondary"]:hover {{ background-color: #252B36 !important; }}
    .stButton button[kind="primary"] {{
        background: linear-gradient(135deg, {COLOR_PURPLE} 0%, #6F4FBF 100%) !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
        padding: 0.35rem !important;
        font-size: 0.85rem !important;
        height: 36px !important;
    }}

    .hero-title {{ display: flex; align-items: center; gap: 0.5rem; margin: 0; }}
    .hero-title h1 {{
        margin: 0 !important;
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        background: linear-gradient(90deg, #C9A0FF 0%, {COLOR_PURPLE} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}
    .hero-subtitle {{
        color: #888A92 !important;
        font-size: 0.72rem !important;
        margin: 0 0 0.3rem 0 !important;
    }}

    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background-color: #161B22 !important;
        border: 1px solid #2D3138 !important;
        border-radius: 10px !important;
        padding: 0.4rem 0.6rem !important;
    }}

    .best-model-box {{
        background: linear-gradient(135deg, #1B2330 0%, #161B22 100%);
        border: 1px solid #2D3138;
        border-radius: 10px;
        padding: 0.45rem 0.7rem;
        height: 100%;
    }}
    .best-model-label {{
        font-size: 0.6rem;
        color: #888A92;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }}
    .best-model-name {{
        font-size: 0.85rem;
        font-weight: 700;
        color: #E8E8EA;
        margin: 0.1rem 0 0;
    }}
    .best-model-f1 {{
        font-size: 1.05rem;
        font-weight: 700;
        color: {COLOR_POS};
        text-align: right;
        margin: 0;
    }}

    .card-num {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 18px;
        height: 18px;
        border-radius: 50%;
        background: linear-gradient(135deg, {COLOR_PURPLE}, #6F4FBF);
        color: white;
        font-size: 0.66rem;
        font-weight: 700;
        margin-right: 0.3rem;
        vertical-align: middle;
    }}
    .card-header-text {{ font-size: 0.82rem; font-weight: 600; color: #E8E8EA; }}

    .sidebar-title {{
        font-size: 0.92rem;
        font-weight: 700;
        color: #C9A0FF;
        line-height: 1.2;
        margin-bottom: 0.05rem;
    }}
    .sidebar-subtitle {{
        color: #888A92 !important;
        font-size: 0.66rem;
        margin-bottom: 0.6rem;
    }}
    .sidebar-section-title {{
        font-size: 0.64rem !important;
        color: #888A92 !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
        margin: 0.55rem 0 0.2rem !important;
    }}

    .model-info-card {{
        background-color: #161B22;
        border: 1px solid #2D3138;
        border-radius: 8px;
        padding: 0.45rem 0.6rem;
        margin: 0.2rem 0 0.45rem;
        font-size: 0.72rem;
    }}
    .model-info-title {{
        font-weight: 600;
        color: #E8E8EA;
        margin-bottom: 0.3rem;
        font-size: 0.76rem;
    }}
    .info-row {{ display: flex; justify-content: space-between; padding: 0.08rem 0; }}
    .info-label {{ color: #888A92; }}
    .info-value {{ color: #E8E8EA; font-weight: 500; }}

    .quick-links {{ font-size: 0.72rem; line-height: 1.6; }}
    .quick-links a {{ color: #C9A0FF; text-decoration: none; }}
    .quick-links a:hover {{ color: #E5C7FF; }}

    .verdict-icon {{ font-size: 2rem; line-height: 1; }}
    .verdict-pos {{ color: {COLOR_POS}; }}
    .verdict-neg {{ color: {COLOR_NEG}; }}

    .proba-bar-track {{
        background-color: #2D3138;
        border-radius: 4px;
        height: 5px;
        overflow: hidden;
        margin-top: 3px;
    }}
    .proba-bar-fill-pos {{
        background: linear-gradient(90deg, {COLOR_POS}, #4DD095);
        height: 100%;
    }}

    .comp-row {{
        display: grid;
        grid-template-columns: 2.4fr 1fr 1fr 0.9fr 0.9fr;
        align-items: center;
        gap: 0.3rem;
        padding: 0.3rem 0.4rem;
        border-bottom: 1px solid #1F232A;
        font-size: 0.74rem;
    }}
    .comp-row.header {{
        color: #888A92;
        font-size: 0.62rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        font-weight: 600;
        border-bottom: 1px solid #2D3138;
    }}
    .winner-f1 {{ color: #C9A0FF !important; font-weight: 700; }}
    .pred-badge-pos {{
        background-color: rgba(46, 194, 126, 0.18);
        color: {COLOR_POS};
        padding: 0.1rem 0.4rem;
        border-radius: 10px;
        font-size: 0.68rem;
        font-weight: 600;
        display: inline-block;
    }}
    .pred-badge-neg {{
        background-color: rgba(229, 72, 77, 0.18);
        color: {COLOR_NEG};
        padding: 0.1rem 0.4rem;
        border-radius: 10px;
        font-size: 0.68rem;
        font-weight: 600;
        display: inline-block;
    }}
    .pred-badge-na {{ color: #555; font-size: 0.68rem; }}

    .tech-panel {{
        background-color: #161B22;
        border: 1px solid #2D3138;
        border-radius: 10px;
        padding: 0.5rem 0.7rem;
        margin-top: 0.3rem;
    }}
    .tech-panel-title {{
        font-size: 0.82rem;
        font-weight: 600;
        color: #E8E8EA;
        margin-bottom: 0.3rem;
    }}
    .tech-grid {{
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 0.6rem;
    }}
    .tech-item-title {{
        font-weight: 600;
        color: #E8E8EA;
        font-size: 0.72rem;
        margin-bottom: 0.15rem;
    }}
    .tech-item-desc {{ color: #B0B3BB; font-size: 0.66rem; line-height: 1.45; }}

    .footer-bar {{
        text-align: center;
        margin-top: 0.3rem;
        padding: 0.25rem 0;
        color: #888A92;
        font-size: 0.68rem;
        border-top: 1px solid #2D3138;
    }}
    .footer-bar a {{ color: #C9A0FF; text-decoration: none; margin: 0 0.4rem; }}

    #MainMenu, footer, header[data-testid="stHeader"] {{ display: none !important; }}
    [data-testid="stToolbar"] {{ display: none !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)


MODELS = {
    "tfidf": {
        "display_name": "TF-IDF + Logistic Regression",
        "type": "Classical ML",
        "size_mb": 8,
        "latency": "~10 ms",
        "f1_val": "91.9 %",
        "params": "243 k",
    },
    "bilstm": {
        "display_name": "BiLSTM + GloVe 300d",
        "type": "Recurrent NN",
        "size_mb": 42,
        "latency": "~150 ms",
        "f1_val": "91.4 %",
        "params": "10.9 M",
    },
    "distilbert": {
        "display_name": "DistilBERT (fine-tuned)",
        "type": "Transformer",
        "size_mb": 253,
        "latency": "~500 ms",
        "f1_val": "93.2 %",
        "params": "66.4 M",
    },
}


COMPARISON_TABLE = [
    {"key": "tfidf", "name": "TF-IDF + Logistic Regression", "f1": "91.9 %", "latency": "~10 ms", "is_winner": False},
    {"key": "bilstm", "name": "BiLSTM + GloVe 300d", "f1": "91.4 %", "latency": "~150 ms", "is_winner": False},
    {"key": "distilbert", "name": "DistilBERT (fine-tuned)", "f1": "93.2 %", "latency": "~500 ms", "is_winner": True},
]


EXAMPLES = {
    "Positive": "The acting was phenomenal, the story kept me engaged from start to finish, and the cinematography was breathtaking. Absolutely one of the best movies I have ever seen!",
    "Negative": "I had high hopes for this one but it ended up being pretty disappointing. The lead actor does his best, but the script is shallow and the editing feels rushed. Not the worst film I've seen, just forgettable.",
    "Sarcastic": "There must be an error. This movie belongs with \"Plan 9\", and a lot others as a quite entertaining, silly diversion. You'll never accept you like it, yet you will watch it whenever it comes out on TV.",
    "Neutral": "The film has its moments. Some scenes are visually impressive and the lead performance shows real talent. However, the screenplay feels underdeveloped and the second act drags considerably.",
}


@st.cache_resource(show_spinner="Loading model from Hugging Face Hub...")
def load_predictor(model_key: str):
    from nlp_sentiment.predict_multi import build_predictor
    return build_predictor(model_key)


def render_gauge(confidence: float, is_positive: bool) -> go.Figure:
    color = COLOR_POS if is_positive else COLOR_NEG

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={"x": [0, 1], "y": [0, 1]},
        number={
            "suffix": "%",
            "font": {"size": 16, "color": "#E8E8EA", "family": "Arial"},
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
        margin={"l": 8, "r": 8, "t": 4, "b": 4},
        height=95,
    )
    return fig


if "textarea_widget" not in st.session_state:
    st.session_state["textarea_widget"] = ""
if "selected_model" not in st.session_state:
    st.session_state["selected_model"] = "tfidf"
if "current_text" not in st.session_state:
    st.session_state["current_text"] = ""
if "predictions_by_model" not in st.session_state:
    st.session_state["predictions_by_model"] = {}
if "last_error" not in st.session_state:
    st.session_state["last_error"] = None


def _reset_predictions_if_text_changed(new_text: str) -> None:
    if new_text != st.session_state["current_text"]:
        st.session_state["predictions_by_model"] = {}
        st.session_state["current_text"] = new_text


def _run_prediction(text: str) -> None:
    _reset_predictions_if_text_changed(text)
    model_key = st.session_state["selected_model"]
    predict_fn, _model, _meta = load_predictor(model_key)
    result = predict_fn(text)
    st.session_state["predictions_by_model"][model_key] = result
    st.session_state["last_error"] = None


def load_example_and_analyze(label: str) -> None:
    text = EXAMPLES[label]
    st.session_state["textarea_widget"] = text
    _run_prediction(text)


def trigger_analyze() -> None:
    text = st.session_state.get("textarea_widget", "").strip()
    if not text:
        st.session_state["predictions_by_model"] = {}
        st.session_state["current_text"] = ""
        st.session_state["last_error"] = "Please enter a review or pick an example."
        return
    _run_prediction(text)


with st.sidebar:
    st.markdown(
        """
        <div class='sidebar-title'>🧠 NLP Sentiment<br/>Analyzer</div>
        <div class='sidebar-subtitle'>Multi-paradigm Analysis</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='sidebar-section-title'>MODEL SELECTION</div>", unsafe_allow_html=True)

    selected_key = st.selectbox(
        label="Model",
        options=list(MODELS.keys()),
        format_func=lambda k: MODELS[k]["display_name"],
        index=list(MODELS.keys()).index(st.session_state["selected_model"]),
        label_visibility="collapsed",
    )
    st.session_state["selected_model"] = selected_key
    selected_info = MODELS[selected_key]

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

    st.markdown("<div class='sidebar-section-title'>QUICK LINKS</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='quick-links'>
        🔗 <a href='https://github.com/sandraFogang/nlp-sentiment-classification' target='_blank'>GitHub</a><br/>
        📖 <a href='https://github.com/sandraFogang/nlp-sentiment-classification#readme' target='_blank'>Documentation</a><br/>
        💼 <a href='https://www.linkedin.com/in/sandrafogang' target='_blank'>LinkedIn</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


hero_col, summary_col = st.columns([3, 1.5])
with hero_col:
    st.markdown(
        """
        <div class='hero-title'>
            <span style='font-size: 1.5rem;'>🎬</span>
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


predictions_by_model = st.session_state["predictions_by_model"]
current_model_result = predictions_by_model.get(st.session_state["selected_model"])
error = st.session_state["last_error"]


input_col, pred_col = st.columns([1, 1])

with input_col:
    with st.container(border=True):
        st.markdown(
            "<span class='card-num'>1</span><span class='card-header-text'>Enter a movie review</span>",
            unsafe_allow_html=True,
        )

        st.text_area(
            label="review",
            height=80,
            placeholder="Type or paste an English movie review here, or pick an example below.",
            label_visibility="collapsed",
            key="textarea_widget",
            max_chars=2000,
        )
        char_count = len(st.session_state["textarea_widget"])
        st.markdown(
            f"<div style='color: #888A92; font-size: 0.66rem; margin: 0.05rem 0;'>"
            f"{char_count} / 2000 characters · Try examples:</div>",
            unsafe_allow_html=True,
        )

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

        if current_model_result:
            label = current_model_result["label"]
            confidence = current_model_result["confidence"]
            proba_pos = current_model_result["probabilities"]["positif"]
            proba_neg = current_model_result["probabilities"]["négatif"]
            is_positive = label == "positif"
            verdict_class = "verdict-pos" if is_positive else "verdict-neg"
            verdict_text = "POSITIVE" if is_positive else "NEGATIVE"
            verdict_icon = "🙂" if is_positive else "🙁"

            v_col, g_col = st.columns([1, 1])
            with v_col:
                st.markdown(
                    f"<div style='text-align: center; padding-top: 0.2rem;'>"
                    f"<div class='verdict-icon {verdict_class}'>{verdict_icon}</div>"
                    f"<div class='{verdict_class}' style='font-size: 1rem; font-weight: 700; margin-top: 0.1rem;'>{verdict_text}</div>"
                    f"<div style='color: #B0B3BB; font-size: 0.64rem; margin-top: 0.15rem;'>"
                    f"{'Positive sentiment.' if is_positive else 'Negative sentiment.'}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with g_col:
                st.plotly_chart(
                    render_gauge(confidence, is_positive),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

            st.markdown(
                f"<div style='display: flex; justify-content: space-between; padding: 0 0.4rem; margin-top: 0.15rem;'>"
                f"<div><div style='color: #B0B3BB; font-size: 0.64rem;'>Positive</div>"
                f"<div style='color: {COLOR_POS}; font-size: 0.9rem; font-weight: 700;'>{proba_pos:.1%}</div></div>"
                f"<div style='text-align: right;'><div style='color: #B0B3BB; font-size: 0.64rem;'>Negative</div>"
                f"<div style='color: {COLOR_NEG}; font-size: 0.9rem; font-weight: 700;'>{proba_neg:.1%}</div></div>"
                f"</div>"
                f"<div style='padding: 0 0.4rem;'>"
                f"<div class='proba-bar-track'><div class='proba-bar-fill-pos' style='width: {proba_pos * 100}%;'></div></div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        elif error:
            st.markdown(
                f"<div style='padding: 0.5rem; color: {COLOR_NEG}; font-size: 0.78rem;'>⚠️ {error}</div>",
                unsafe_allow_html=True,
            )
        elif predictions_by_model:
            st.markdown(
                f"<div style='text-align: center; padding: 0.7rem 0.5rem; color: #888A92;'>"
                f"<div style='font-size: 1.4rem;'>🔄</div>"
                f"<p style='margin-top: 0.2rem; font-size: 0.72rem;'>Click <strong>Analyze Sentiment</strong> "
                f"to compute the prediction with <strong>{selected_info['display_name']}</strong>.</p>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='text-align: center; padding: 0.7rem 0.5rem; color: #888A92;'>"
                "<div style='font-size: 1.4rem;'>📊</div>"
                "<p style='margin-top: 0.2rem; font-size: 0.72rem;'>Awaiting input… "
                "Pick an example or type a review.</p>"
                "</div>",
                unsafe_allow_html=True,
            )


interp_col, comp_col = st.columns([1, 1])

with interp_col:
    with st.container(border=True):
        st.markdown(
            "<span class='card-num'>3</span><span class='card-header-text'>Why this prediction?</span> "
            "<span style='color: #888A92; font-size: 0.68rem;'>(Interpretability)</span>",
            unsafe_allow_html=True,
        )

        if current_model_result:
            from nlp_sentiment.interpretability import explain_tfidf, explain_by_occlusion

            try:
                model_key = st.session_state["selected_model"]
                predict_fn, model_obj, meta = load_predictor(model_key)
                analyzed_text = st.session_state["current_text"]

                if model_key == "tfidf":
                    pos_words, neg_words = explain_tfidf(
                        analyzed_text, model_obj, meta, top_k=5,
                    )
                else:
                    spinner_msg = (
                        "Computing word importance via occlusion (~5s)..."
                        if model_key == "bilstm"
                        else "Computing word importance via occlusion (~30s)..."
                    )
                    with st.spinner(spinner_msg):
                        pos_words, neg_words = explain_by_occlusion(
                            analyzed_text, predict_fn, top_k=5,
                        )

                tab_pos, tab_neg = st.tabs(["Positive", "Negative"])

                def render_words(words, color):
                    if not words:
                        st.caption("No contribution detected.")
                        return
                    max_c = max(w["contribution"] for w in words)
                    for w in words:
                        bar = (w["contribution"] / max_c) * 100
                        st.markdown(
                            f"<div style='display: flex; align-items: center; gap: 0.3rem; margin: 0.15rem 0;'>"
                            f"<div style='width: 80px; color: #E0E0E0; font-weight: 600; font-size: 0.72rem;'>{w['word']}</div>"
                            f"<div style='flex: 1; background-color: #2D3138; border-radius: 4px; height: 4px;'>"
                            f"<div style='width: {bar}%; background: {color}; height: 100%; border-radius: 4px;'></div></div>"
                            f"<div style='width: 32px; text-align: right; color: {color}; font-weight: 600; font-size: 0.68rem;'>{w['contribution']:.2f}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                with tab_pos:
                    render_words(pos_words, COLOR_POS)
                with tab_neg:
                    render_words(neg_words, COLOR_NEG)

            except Exception as exc:
                st.error(f"Interpretability error: {exc}")
        else:
            st.markdown(
                "<div style='text-align: center; padding: 0.7rem 0.5rem; color: #888A92;'>"
                "<div style='font-size: 1.3rem;'>🔍</div>"
                "<p style='margin-top: 0.2rem; font-size: 0.72rem;'>Awaiting input… "
                "Run an analysis to see word contributions.</p>"
                "</div>",
                unsafe_allow_html=True,
            )


with comp_col:
    with st.container(border=True):
        st.markdown(
            "<span class='card-num'>4</span><span class='card-header-text'>Compare with other models</span> "
            "<span style='color: #888A92; font-size: 0.68rem;'>(Validation Set)</span>",
            unsafe_allow_html=True,
        )

        html_parts = ["<div class='comp-row header'><div>Model</div><div>Prediction</div><div>Confidence</div><div>F1 (Val)</div><div>Latency</div></div>"]

        for row in COMPARISON_TABLE:
            if row["key"] in predictions_by_model:
                pred = predictions_by_model[row["key"]]
                pred_label = "Positive" if pred["label"] == "positif" else "Negative"
                badge = (
                    f"<span class='pred-badge-pos'>{pred_label}</span>"
                    if pred_label == "Positive"
                    else f"<span class='pred-badge-neg'>{pred_label}</span>"
                )
                conf = f"{pred['confidence']:.1%}"
            else:
                badge = "<span class='pred-badge-na'>—</span>"
                conf = "<span class='pred-badge-na'>—</span>"

            f1_class = "winner-f1" if row["is_winner"] else ""
            winner_marker = " 🏆" if row["is_winner"] else ""

            html_parts.append(
                f"<div class='comp-row'><div style='color:#E0E0E0;'>{row['name']}{winner_marker}</div>"
                f"<div>{badge}</div><div>{conf}</div>"
                f"<div class='{f1_class}'>{row['f1']}</div>"
                f"<div style='color:#B0B3BB;'>{row['latency']}</div></div>"
            )

        st.markdown("".join(html_parts), unsafe_allow_html=True)

        st.markdown(
            "<div style='color: #888A92; font-size: 0.6rem; margin-top: 0.25rem;'>"
            "ⓘ F1 macro on validation (3 000 reviews). Predictions accumulate as you switch models on the same review."
            "</div>",
            unsafe_allow_html=True,
        )


st.markdown(
    """
    <div class='tech-panel'>
        <div class='tech-panel-title'>📋 Technical Details &amp; Model Card</div>
        <div class='tech-grid'>
            <div><div class='tech-item-title'>📂 Dataset</div><div class='tech-item-desc'>50,000 IMDB movie reviews (Maas et al., 2011), split into 22k for training, 3k for validation, and 25k for the held-out test set.</div></div>
            <div><div class='tech-item-title'>⚙️ Preprocessing</div><div class='tech-item-desc'>Lowercasing, punctuation removal and tokenization for the classical and recurrent models. WordPiece tokenization for DistilBERT, which keeps subwords and special tokens.</div></div>
            <div><div class='tech-item-title'>🧬 Models Evaluated</div><div class='tech-item-desc'>Three paradigms hosted on the Hugging Face Hub and loaded on demand: TF-IDF with logistic regression, BiLSTM initialized with GloVe 300d embeddings, and DistilBERT fine-tuned end-to-end.</div></div>
            <div><div class='tech-item-title'>📊 Metrics</div><div class='tech-item-desc'>Macro F1 score (the main selection criterion), accuracy, precision, recall, and latency per inference. The validation set drives model selection; final numbers come from the test set.</div></div>
            <div><div class='tech-item-title'>⚠️ Limitations</div><div class='tech-item-desc'>The models work only on English movie reviews and tend to fail on sarcasm and short texts. Predictions reflect the IMDB corpus from 2011 and should not be used for automated decisions without human review.</div></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


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