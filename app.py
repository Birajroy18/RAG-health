import streamlit as st
from html import escape
from streamlit.errors import StreamlitSecretNotFoundError

from load_secrets import get_secret, init_secrets
from rag_pipeline import OpenRouterBusyError, rag_query

init_secrets()

st.set_page_config(page_title="Healthcare Assistant", page_icon="🩺", layout="wide")


def get_configured_api_key(key: str) -> str:
    try:
        secret_key = st.secrets.get(key)
        if secret_key:
            return str(secret_key).strip()
    except StreamlitSecretNotFoundError:
        pass
    return get_secret(key)


def _truncate(text: str, max_len: int = 72) -> str:
    text = " ".join(text.split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


st.markdown(
    """
<style>
    .main { background-color: #f0f4f8; }
    .stApp { font-family: 'Segoe UI', sans-serif; }
    .source-box {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 10px;
        font-size: 0.88rem;
        color: #1e293b;
    }
    .answer-box {
        background: #ecfdf5;
        border-left: 4px solid #10b981;
        padding: 16px 20px;
        border-radius: 8px;
        font-size: 1rem;
        color: #0f172a;
    }
    .no-answer-box {
        background: #fef2f2;
        border-left: 4px solid #ef4444;
        padding: 16px 20px;
        border-radius: 8px;
        font-size: 1rem;
        color: #7f1d1d;
    }
    .meta-tag {
        font-size: 0.75rem;
        color: #64748b;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .sidebar-question {
        font-size: 0.9rem;
        color: #334155;
        margin-bottom: 0.65rem;
        line-height: 1.35;
    }
    [data-testid="stSidebar"] p.medai-brand {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        color: #059669 !important;
        letter-spacing: 0.04em !important;
        margin: 0 0 0.35rem 0 !important;
        line-height: 1.15 !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# In-memory only for this browser session; cleared on page reload (new Streamlit session).
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

gemini_api_key = get_configured_api_key("GEMINI_API_KEY")
openrouter_api_key = get_configured_api_key("OPENROUTER_API_KEY")
has_llm_key = bool(gemini_api_key or openrouter_api_key)

with st.sidebar:
    st.markdown(
        '<p class="medai-brand" style="font-size:2.5rem;font-weight:800;color:#059669;'
        'letter-spacing:0.04em;margin:0 0 0.35rem 0;line-height:1.15;">MedAI</p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    if not gemini_api_key:
        gemini_api_key = st.text_input(
            "Gemini API Key",
            type="password",
            placeholder="AIza...",
            help="Preferred for query cleanup and grounded answer generation.",
        )

    if not openrouter_api_key:
        openrouter_api_key = st.text_input(
            "OpenRouter API Key (fallback)",
            type="password",
            placeholder="sk-or-...",
            help="Optional fallback if Gemini is unavailable.",
        )
        st.markdown("---")

    if st.session_state.chat_history:
        for index, chat in enumerate(st.session_state.chat_history, start=1):
            st.markdown(
                f'<p class="sidebar-question"><strong>{index}.</strong> '
                f'{_truncate(chat["query"])}</p>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("Questions you ask will appear here.")

    st.markdown("---")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

st.title("Healthcare Assistant")
st.caption(
    "Enter your symptoms and let MedAI analyse them using live medical literature "
    "it can also answer questions related to drugs, diseases symptoms — all for educational awareness."
    "MedAI does not replace a licensed medical professional."
)

if not (gemini_api_key or openrouter_api_key):
    st.warning("Add your **Gemini API key** or **OpenRouter API key** in `secrets.toml` or the sidebar.")

example_cols = st.columns(2)
with example_cols[0]:
    st.markdown("**How questions should be asked for best results:**")
    st.markdown("- Latest research on type 2 diabetes")
    st.markdown("- What are the treatment options for fever?")



def render_sources(sources: list[dict]) -> None:
    for src in sources:
        link = f'<a href="{src["url"]}" target="_blank">{src["url"]}</a>'
        st.markdown(
            f'<div class="source-box">'
            f'<div class="meta-tag">'
            f'[{src["rank"]}] {src["source"]} | {src["source_id"]} | '
            f'relevance {src.get("relevance_score", "—")}'
            f"</div>"
            f'<strong>{src["title"]}</strong><br>{link}<br><br>'
            f'{src["text"][:1200]}{"…" if len(src["text"]) > 1200 else ""}'
            f"</div>",
            unsafe_allow_html=True,
        )


def render_answer(answer: str, grounded: bool) -> None:
    css_class = "answer-box" if grounded else "no-answer-box"
    formatted = escape(answer).replace("\n", "<br>")
    st.markdown(f'<div class="{css_class}">{formatted}</div>', unsafe_allow_html=True)


for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(chat["query"])
    with st.chat_message("assistant"):
        render_answer(chat["answer"], chat.get("grounded", True))
        if chat.get("sources"):
            with st.expander(f"Sources ({len(chat['sources'])})"):
                render_sources(chat["sources"])

query = st.chat_input(
    "Ask about symptoms, diseases, drugs, or latest research...",
    disabled=not bool(gemini_api_key or openrouter_api_key),
)

if query and (gemini_api_key or openrouter_api_key):
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching medical databases..."):
            try:
                result = rag_query(query, openrouter_api_key, gemini_api_key)
                render_answer(result["answer"], result.get("grounded", False))
                if result.get("sources"):
                    with st.expander(f"Sources ({len(result['sources'])})"):
                        render_sources(result["sources"])
                st.session_state.chat_history.append(
                    {
                        "query": query,
                        "answer": result["answer"],
                        "sources": result.get("sources", []),
                        "grounded": result.get("grounded", False),
                    }
                )
                st.rerun()
            except OpenRouterBusyError as exc:
                st.warning(str(exc))
            except Exception as exc:
                st.error(f"Error: {exc}")
