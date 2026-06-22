# app/ui/streamlit_app.py — Interface RAG Omnishore avec pipeline animé

import streamlit as st
import time
import os
from pathlib import Path
from docling.document_converter import DocumentConverter
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import ollama

# ─────────────────────────────────────────────────────────────
# CONFIG PAGE
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Omnishore — RAG Lab",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

DOCS_DIR = Path("documents")
DOCS_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────
# CSS CUSTOM
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
    --bg-deep: #1A1B2E;
    --bg-panel: #22243B;
    --bg-panel-light: #2A2D4A;
    --coral: #FF6B6B;
    --turquoise: #4ECDC4;
    --yellow: #FFE66D;
    --violet: #A78BFA;
    --mint: #06FFA5;
    --text-main: #F0F0F5;
    --text-dim: #8B8DA8;
}

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

.stApp {
    background: var(--bg-deep);
    background-image:
        radial-gradient(circle at 15% 20%, rgba(255,107,107,0.07) 0%, transparent 40%),
        radial-gradient(circle at 85% 80%, rgba(167,139,250,0.07) 0%, transparent 40%);
}
#MainMenu, footer, header {visibility: hidden;}

/* ── Hero / branding Omnishore ── */
.lab-hero { padding: 1.2rem 0 0.5rem 0; }
.brand-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.8rem;
}
.brand-mark {
    width: 38px;
    height: 38px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--coral), var(--violet));
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 1rem;
    color: #fff;
    flex-shrink: 0;
}
.brand-name {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text-main);
    letter-spacing: 0.02em;
}
.brand-tagline {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    color: var(--mint);
    text-transform: uppercase;
}
.lab-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    color: var(--mint);
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.lab-title {
    font-size: 2.6rem;
    font-weight: 700;
    color: var(--text-main);
    line-height: 1.1;
    margin: 0;
}
.lab-title span {
    background: linear-gradient(90deg, var(--coral), var(--violet));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.lab-sub { color: var(--text-dim); font-size: 1rem; margin-top: 0.5rem; }

/* ── Pipeline stations — cercles ── */
.pipeline-wrap {
    display: flex;
    align-items: flex-start;
    gap: 0;
    margin: 2rem 0 1.5rem 0;
    overflow-x: auto;
    padding: 0.6rem 0 1rem 0;
}
.station {
    flex: 0 0 auto;
    width: 130px;
    height: 130px;
    border-radius: 50%;
    background: var(--bg-panel);
    border: 2px solid rgba(255,255,255,0.06);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
}
.station-icon { font-size: 1.5rem; margin-bottom: 0.25rem; display: block; }
.station-label {
    font-size: 0.62rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--text-dim);
}
.station-time {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: var(--text-main);
    margin-top: 0.2rem;
    min-height: 1.1rem;
}
.connector {
    width: 22px;
    height: 2px;
    background: rgba(255,255,255,0.08);
    flex-shrink: 0;
    margin-top: 65px;
}
.station.active {
    transform: scale(1.08);
    box-shadow: 0 0 0 6px var(--glow-color, rgba(255,107,107,0.15)), 0 8px 24px var(--glow-color, rgba(255,107,107,0.4));
    animation: pulse 1.2s ease-in-out infinite;
}
.station.done { border-color: var(--glow-color, var(--mint)); background: var(--bg-panel-light); }
.station.idle { opacity: 0.35; }

@keyframes pulse {
    0%, 100% { box-shadow: 0 0 0 6px var(--glow-color, rgba(255,107,107,0.15)), 0 8px 24px var(--glow-color, rgba(255,107,107,0.4)); }
    50% { box-shadow: 0 0 0 10px var(--glow-color, rgba(255,107,107,0.1)), 0 8px 36px var(--glow-color, rgba(255,107,107,0.7)); }
}

.s-extract { --glow-color: rgba(255,107,107,0.5); }
.s-extract.done, .s-extract.active { border-color: var(--coral); }
.s-embed { --glow-color: rgba(78,205,196,0.5); }
.s-embed.done, .s-embed.active { border-color: var(--turquoise); }
.s-qdrant { --glow-color: rgba(255,230,109,0.5); }
.s-qdrant.done, .s-qdrant.active { border-color: var(--yellow); }
.s-rerank { --glow-color: rgba(167,139,250,0.5); }
.s-rerank.done, .s-rerank.active { border-color: var(--violet); }
.s-llm { --glow-color: rgba(6,255,165,0.5); }
.s-llm.done, .s-llm.active { border-color: var(--mint); }

/* ── Answer "medallion" ── */
.answer-medallion {
    background: var(--bg-panel);
    border-radius: 32px;
    border: 2px solid rgba(6,255,165,0.25);
    padding: 2rem 2.2rem;
    margin-top: 1rem;
    position: relative;
}
.answer-medallion::before {
    content: "💡";
    position: absolute;
    top: -18px;
    left: 32px;
    background: var(--mint);
    width: 36px;
    height: 36px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
}
.answer-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    color: var(--mint);
    text-transform: uppercase;
    margin-bottom: 0.8rem;
    margin-left: 0.2rem;
}
.answer-text { color: var(--text-main); font-size: 1.05rem; line-height: 1.6; }

/* ── stat pills (cercles) ── */
.stat-row { display: flex; gap: 0.9rem; margin-top: 1.3rem; flex-wrap: wrap; }
.stat-pill {
    background: var(--bg-panel-light);
    border-radius: 50px;
    padding: 0.55rem 1.1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.76rem;
    color: var(--text-dim);
    border: 1px solid rgba(255,255,255,0.06);
}
.stat-pill b { color: var(--text-main); }

/* ── Source bubbles ── */
.source-bubble {
    background: var(--bg-panel);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 28px;
    padding: 1.3rem 1.6rem;
    margin-bottom: 0.8rem;
    display: flex;
    gap: 1.2rem;
    align-items: flex-start;
}
.source-num {
    flex-shrink: 0;
    width: 46px;
    height: 46px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--violet), var(--coral));
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    color: #fff;
    font-size: 1rem;
}
.source-body { flex: 1; }
.source-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem; }
.source-tag { font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: var(--violet); font-weight: 700; }
.source-score {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--text-dim);
    background: rgba(167,139,250,0.12);
    padding: 0.15rem 0.6rem;
    border-radius: 50px;
}
.source-snippet { color: var(--text-dim); font-size: 0.87rem; line-height: 1.5; }

/* ── doc chip in sidebar ── */
.doc-chip {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    background: var(--bg-panel-light);
    border-radius: 50px;
    padding: 0.5rem 1rem;
    margin-top: 0.5rem;
    font-size: 0.8rem;
}
.doc-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--mint); flex-shrink: 0; }

/* ── Input styling ── */
.stTextInput input {
    background: var(--bg-panel) !important;
    color: var(--text-main) !important;
    border: 2px solid rgba(255,255,255,0.08) !important;
    border-radius: 50px !important;
    font-size: 1rem !important;
    padding: 0.8rem 1.4rem !important;
}
.stTextInput input:focus {
    border-color: var(--violet) !important;
    box-shadow: 0 0 0 3px rgba(167,139,250,0.15) !important;
}
.stButton button {
    background: linear-gradient(90deg, var(--coral), var(--violet)) !important;
    color: white !important;
    border: none !important;
    border-radius: 50px !important;
    font-weight: 600 !important;
    padding: 0.7rem 1.8rem !important;
    transition: transform 0.2s !important;
}
.stButton button:hover { transform: scale(1.03); }

[data-testid="stFileUploader"] {
    background: var(--bg-panel);
    border-radius: 24px;
    padding: 1rem;
    border: 2px dashed rgba(255,255,255,0.12);
}
[data-testid="stFileUploader"] section { background: transparent; }

section[data-testid="stSidebar"] { background: var(--bg-panel); }
section[data-testid="stSidebar"] * { color: var(--text-main) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CACHE — modèles (ce qui le rend indépendant pas du document)
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("BAAI/bge-m3")
    reranker = CrossEncoder("BAAI/bge-reranker-base")
    client = QdrantClient(host="localhost", port=6333)
    return embedder, reranker, client

# ─────────────────────────────────────────────────────────────
# SIDEBAR — config + upload
# ─────────────────────────────────────────────────────────────
if "indexed_doc" not in st.session_state:
    st.session_state.indexed_doc = None
if "n_chunks" not in st.session_state:
    st.session_state.n_chunks = 0
if "indexation_time" not in st.session_state:
    st.session_state.indexation_time = None

with st.sidebar:
    st.markdown("### 📁 Document source")

    uploaded = st.file_uploader("Glisse un PDF à indexer", type=["pdf"], label_visibility="collapsed")

    if uploaded is not None:
        save_path = DOCS_DIR / uploaded.name
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())

        if st.button("🔄 Indexer ce document", use_container_width=True):
            embedder, reranker, client = load_models()
            progress = st.progress(0, text="Extraction du PDF...")
            t_index_start = time.time()

            converter = DocumentConverter()
            result = converter.convert(str(save_path))
            text = result.document.export_to_markdown()
            progress.progress(30, text="Chunking...")

            llama_doc = Document(text=text)
            splitter = SentenceSplitter(chunk_size=512, chunk_overlap=100)
            chunks = splitter.get_nodes_from_documents([llama_doc])
            texts = [c.text for c in chunks]
            progress.progress(55, text=f"Embeddings ({len(texts)} chunks)...")

            embeddings = embedder.encode(texts, show_progress_bar=False)
            progress.progress(80, text="Indexation Qdrant...")

            if client.collection_exists("documents"):
                client.delete_collection("documents")
            client.create_collection(
                collection_name="documents",
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
            )
            points = [
                PointStruct(id=i, vector=embeddings[i].tolist(), payload={"text": texts[i]})
                for i in range(len(texts))
            ]
            client.upsert(collection_name="documents", points=points)
            progress.progress(100, text="Terminé !")

            indexation_time = time.time() - t_index_start

            st.session_state.indexed_doc = uploaded.name
            st.session_state.n_chunks = len(texts)
            st.session_state.indexation_time = indexation_time
            st.success(f"✅ {len(texts)} chunks indexés en {indexation_time:.1f}s")
            time.sleep(0.5)
            st.rerun()

    if st.session_state.indexed_doc:
        time_str = f" · {st.session_state.indexation_time:.1f}s" if st.session_state.indexation_time else ""
        st.markdown(f"""
        <div class="doc-chip">
            <span class="doc-dot"></span>
            <span><b>{st.session_state.indexed_doc}</b><br>
            <span style="font-size:0.7rem; color:#8B8DA8;">{st.session_state.n_chunks} chunks{time_str}</span></span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="font-size: 0.78rem; color: #8B8DA8; margin-top: 0.5rem;">
        Aucun document indexé pour cette session — uploade un PDF ci-dessus.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚙️ Configuration active")
    st.markdown("""
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; line-height: 2;">
    <b>Embeddings</b><br>BAAI/bge-m3<br><br>
    <b>Reranker</b><br>BAAI/bge-reranker-base<br><br>
    <b>LLM</b><br>mistral (Ollama, local)<br><br>
    <b>Retrieval</b><br>top_k = 5 → top_n = 2 × 500 chars
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.78rem; color: #8B8DA8;">
    Pipeline 100% local. Aucune donnée n'est envoyée à un service externe.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# HERO — branding Omnishore
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="lab-hero">
    <div class="brand-row">
        <div class="brand-mark">OM</div>
        <div>
            <div class="brand-name">Omnishore <span style="color:#8B8DA8; font-weight:400;">— Groupe Medtech</span></div>
            <div class="brand-tagline">Architecte de votre transformation digitale</div>
        </div>
    </div>
    <h1 class="lab-title">Pose ta question<br><span>à tes documents</span></h1>
    <p class="lab-sub">Extraction → Embedding → Qdrant → Reranking → Mistral — pipeline RAG 100% local et vérifiable.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# PIPELINE VISUAL
# ─────────────────────────────────────────────────────────────
STATIONS = [
    {"key": "extract", "icon": "📄", "label": "Extraction", "class": "s-extract"},
    {"key": "embed",   "icon": "🔢", "label": "Embedding",  "class": "s-embed"},
    {"key": "qdrant",  "icon": "💾", "label": "Qdrant",     "class": "s-qdrant"},
    {"key": "rerank",  "icon": "📊", "label": "Reranking",  "class": "s-rerank"},
    {"key": "llm",     "icon": "🤖", "label": "Mistral",    "class": "s-llm"},
]

def render_pipeline(states, timings):
    html = '<div class="pipeline-wrap">'
    for i, s in enumerate(STATIONS):
        state = states.get(s["key"], "idle")
        t = timings.get(s["key"], "")
        time_html = f"{t:.2f}s" if isinstance(t, float) else "—"
        html += f"""
        <div class="station {s['class']} {state}">
            <span class="station-icon">{s['icon']}</span>
            <div class="station-label">{s['label']}</div>
            <div class="station-time">{time_html}</div>
        </div>
        """
        if i < len(STATIONS) - 1:
            html += '<div class="connector"></div>'
    html += '</div>'
    return html

pipeline_slot = st.empty()
pipeline_slot.markdown(render_pipeline({k["key"]: "idle" for k in STATIONS}, {}), unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# INPUT
# ─────────────────────────────────────────────────────────────
col1, col2 = st.columns([5, 1])
with col1:
    query = st.text_input("Question", placeholder="Ex: What is the sickness policy?", label_visibility="collapsed")
with col2:
    ask = st.button("Rechercher →", use_container_width=True)

result_slot = st.empty()

# ─────────────────────────────────────────────────────────────
# PIPELINE EXECUTION
# ─────────────────────────────────────────────────────────────
if ask and not st.session_state.indexed_doc:
    st.warning("Indexe d'abord un document via la barre latérale 📁")

elif ask and query:
    embedder, reranker, client = load_models()
    states = {k["key"]: "idle" for k in STATIONS}
    timings = {}

    states["extract"] = "done"
    timings["extract"] = 0.0
    pipeline_slot.markdown(render_pipeline(states, timings), unsafe_allow_html=True)

    states["embed"] = "active"
    pipeline_slot.markdown(render_pipeline(states, timings), unsafe_allow_html=True)
    t0 = time.time()
    query_vector = embedder.encode(query).tolist()
    timings["embed"] = time.time() - t0
    states["embed"] = "done"
    pipeline_slot.markdown(render_pipeline(states, timings), unsafe_allow_html=True)

    states["qdrant"] = "active"
    pipeline_slot.markdown(render_pipeline(states, timings), unsafe_allow_html=True)
    t0 = time.time()
    results = client.query_points(collection_name="documents", query=query_vector, limit=5)
    timings["qdrant"] = time.time() - t0
    states["qdrant"] = "done"
    pipeline_slot.markdown(render_pipeline(states, timings), unsafe_allow_html=True)

    states["rerank"] = "active"
    pipeline_slot.markdown(render_pipeline(states, timings), unsafe_allow_html=True)
    t0 = time.time()
    pairs = [(query, r.payload["text"]) for r in results.points]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(results.points, scores), key=lambda x: x[1], reverse=True)[:2]
    timings["rerank"] = time.time() - t0
    states["rerank"] = "done"
    pipeline_slot.markdown(render_pipeline(states, timings), unsafe_allow_html=True)

    states["llm"] = "active"
    pipeline_slot.markdown(render_pipeline(states, timings), unsafe_allow_html=True)
    t0 = time.time()

    context = "\n\n".join([
        f"[Source {i+1}]:\n{point.payload['text'][:500]}"
        for i, (point, score) in enumerate(ranked)
    ])
    prompt = f"""You are a helpful assistant. Answer the question based ONLY on the sources below.
If the answer is not in the sources, say "I don't have enough information."
Always cite which source you used.

{context}

Question: {query}
Answer:"""

    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
    timings["llm"] = time.time() - t0
    states["llm"] = "done"
    pipeline_slot.markdown(render_pipeline(states, timings), unsafe_allow_html=True)

    answer = response["message"]["content"]
    total_time = sum(timings.values())

    with result_slot.container():
        st.markdown(f"""
        <div class="answer-medallion">
            <div class="answer-label">Réponse</div>
            <div class="answer-text">{answer}</div>
            <div class="stat-row">
                <div class="stat-pill">⏱️ Total <b>{total_time:.1f}s</b></div>
                <div class="stat-pill">📥 Prompt <b>{response.get('prompt_eval_count', '—')} tokens</b></div>
                <div class="stat-pill">📤 Généré <b>{response.get('eval_count', '—')} tokens</b></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### 📚 Sources utilisées")

        for i, (point, score) in enumerate(ranked):
            st.markdown(f"""
            <div class="source-bubble">
                <div class="source-num">{i+1}</div>
                <div class="source-body">
                    <div class="source-head">
                        <span class="source-tag">SOURCE {i+1}</span>
                        <span class="source-score">score {score:.3f}</span>
                    </div>
                    <div class="source-snippet">{point.payload['text'][:280]}...</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

elif ask and not query:
    st.warning("Tape une question avant de lancer la recherche 🙂")