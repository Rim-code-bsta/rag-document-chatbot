# app/ui/streamlit_app.py — RAG Omnishore · version avec gestion des versions de documents

import streamlit as st
import time
import sys
import math
from datetime import datetime
from pathlib import Path
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Document
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchAny,
)
import ollama

st.set_page_config(
    page_title="Omnishore — RAG Lab",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

DOCS_DIR = Path("documents")
DOCS_DIR.mkdir(exist_ok=True)

#  KEYWORD MAP 
KEYWORD_MAP = {
    "historique":   ["historique", "2015", "2016", "2017", "2018", "2019", "2020",
                     "fondation", "fondée", "fondé", "creation", "créée", "crée",
                     "histoire", "depuis", "ans", "origine", "début"],
    "congé":        ["conge", "conges", "vacances", "absence", "repos"],
    "salaire":      ["salaire", "remuneration", "paie", "compensation", "augmentation"],
    "formation":    ["formation", "apprentissage", "training", "competence"],
    "recrutement":  ["recrutement", "embauche", "candidat", "offre emploi"],
    "depense":      ["depense", "frais", "remboursement", "note de frais"],
    "hebergement":  ["hebergement", "hotel", "nuitee", "logement"],
    "repas":        ["repas", "restauration", "dejeuner", "diner", "plafond"],
    "transport":    ["transport", "deplacement", "billet", "train", "avion"],
    "objectif":     ["objectif", "performance", "evaluation", "kpi"],
    "discipline":   ["discipline", "sanction", "avertissement", "licenciement"],
    "vision":       ["vision", "mission", "strategie", "valeurs"],
}


#  VERSIONING HELPERS 
# Each entry in st.session_state.indexed_docs is keyed by filename and holds:
#   {
#     "versions": [
#       { "version_id": "rapport_rh::v1", "label": "v1", "n_chunks": 42,
#         "time": 3.2, "timestamp": "2024-06-27 14:32" },
#       ...
#     ],
#     "active_versions": ["rapport_rh::v1", "rapport_rh::v2"],  # which to query
#   }
#
# version_id format: "<filename_stem>::v<N>"
# This is stored in each Qdrant point's payload under the key "version_id",
# allowing Qdrant's must-filter to restrict retrieval to selected versions only.

def make_version_id(filename: str, version_num: int) -> str:
    stem = Path(filename).stem
    return f"{stem}::v{version_num}"


def next_version_number(filename: str) -> int:
    doc = st.session_state.indexed_docs.get(filename)
    if not doc:
        return 1
    return len(doc["versions"]) + 1


def get_all_active_version_ids() -> list[str]:
    """
    Returns the list of version_ids currently selected for querying,
    across all documents. Empty list means no filter (should not happen
    if at least one doc is indexed).
    """
    ids = []
    for doc in st.session_state.indexed_docs.values():
        ids.extend(doc.get("active_versions", []))
    return ids


#  KEYWORD DETECTION 
def detect_keywords(query: str) -> list[str]:
    query_lower = query.lower()
    matched = []
    seen = set()
    for trigger, keywords in KEYWORD_MAP.items():
        if trigger in query_lower or any(kw in query_lower for kw in keywords):
            if trigger not in seen:
                matched.extend(keywords)
                seen.add(trigger)
    return matched


def keyword_fallback(client, query, existing_points, version_filter: Filter | None):
    matched_keywords = detect_keywords(query)
    if not matched_keywords:
        return []

    all_chunks = client.scroll(
        collection_name="documents",
        limit=1000,
        with_payload=True,
        with_vectors=False,
        scroll_filter=version_filter,   # ← respect version selection
    )[0]

    existing_ids = {p.id for p in existing_points}
    extra = []
    for point in all_chunks:
        text_lower = point.payload.get("text", "").lower()
        if any(kw in text_lower for kw in matched_keywords):
            if point.id not in existing_ids:
                extra.append(point)
                existing_ids.add(point.id)

    print(f"[KEYWORD FALLBACK] {len(extra)} chunks supplémentaires", file=sys.stderr)
    return extra[:10]


#  TABLE→TEXT CONVERTER 
def _markdown_tables_to_text(md: str) -> str:
    import re
    lines = md.split("\n")
    output = []
    headers: list[str] = []
    in_table = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped.split("|") if c.strip()]
            if all(re.match(r"^[-:]+$", c) for c in cells):
                in_table = True
                continue
            if not in_table and not headers:
                headers = cells
                in_table = True
            else:
                if headers and len(cells) == len(headers):
                    output.append(" | ".join(f"{h}: {v}" for h, v in zip(headers, cells)))
                else:
                    output.append("  ".join(cells))
        else:
            if in_table:
                in_table = False
                headers = []
            output.append(line)
    return "\n".join(output)


#  SIGMOID 
def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


#  CSS 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');
:root {
    --bg-deep: #1A1B2E; --bg-panel: #22243B; --bg-panel-light: #2A2D4A;
    --coral: #FF6B6B; --turquoise: #4ECDC4; --yellow: #FFE66D;
    --violet: #A78BFA; --mint: #06FFA5; --text-main: #F0F0F5; --text-dim: #8B8DA8;
}
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.stApp {
    background: var(--bg-deep);
    background-image: radial-gradient(circle at 15% 20%, rgba(255,107,107,0.07) 0%, transparent 40%),
                      radial-gradient(circle at 85% 80%, rgba(167,139,250,0.07) 0%, transparent 40%);
}
#MainMenu, footer, header {visibility: hidden;}
.lab-hero { padding: 1.2rem 0 0.5rem 0; }
.brand-row { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.8rem; }
.brand-mark { width: 38px; height: 38px; border-radius: 50%; background: linear-gradient(135deg, var(--coral), var(--violet)); display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 1rem; color: #fff; flex-shrink: 0; }
.brand-name { font-size: 1.1rem; font-weight: 700; color: var(--text-main); letter-spacing: 0.02em; }
.brand-tagline { font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; letter-spacing: 0.12em; color: var(--mint); text-transform: uppercase; }
.lab-title { font-size: 2.6rem; font-weight: 700; color: var(--text-main); line-height: 1.1; margin: 0; }
.lab-title span { background: linear-gradient(90deg, var(--coral), var(--violet)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.lab-sub { color: var(--text-dim); font-size: 1rem; margin-top: 0.5rem; }
.pipeline-wrap { display: flex; align-items: flex-start; gap: 0; margin: 2rem 0 1.5rem 0; overflow-x: auto; padding: 0.6rem 0 1rem 0; }
.station { flex: 0 0 auto; width: 130px; height: 130px; border-radius: 50%; background: var(--bg-panel); border: 2px solid rgba(255,255,255,0.06); display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1); position: relative; }
.station-icon { font-size: 1.5rem; margin-bottom: 0.25rem; display: block; }
.station-label { font-size: 0.62rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; color: var(--text-dim); }
.station-time { font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: var(--text-main); margin-top: 0.2rem; min-height: 1.1rem; }
.connector { width: 22px; height: 2px; background: rgba(255,255,255,0.08); flex-shrink: 0; margin-top: 65px; }
.station.active { transform: scale(1.08); box-shadow: 0 0 0 6px var(--glow-color, rgba(255,107,107,0.15)), 0 8px 24px var(--glow-color, rgba(255,107,107,0.4)); animation: pulse 1.2s ease-in-out infinite; }
.station.done { border-color: var(--glow-color, var(--mint)); background: var(--bg-panel-light); }
.station.idle { opacity: 0.35; }
@keyframes pulse {
    0%, 100% { box-shadow: 0 0 0 6px var(--glow-color, rgba(255,107,107,0.15)), 0 8px 24px var(--glow-color, rgba(255,107,107,0.4)); }
    50%       { box-shadow: 0 0 0 10px var(--glow-color, rgba(255,107,107,0.1)), 0 8px 36px var(--glow-color, rgba(255,107,107,0.7)); }
}
.s-extract { --glow-color: rgba(255,107,107,0.5); } .s-extract.done, .s-extract.active { border-color: var(--coral); }
.s-embed   { --glow-color: rgba(78,205,196,0.5);  } .s-embed.done,   .s-embed.active   { border-color: var(--turquoise); }
.s-qdrant  { --glow-color: rgba(255,230,109,0.5); } .s-qdrant.done,  .s-qdrant.active  { border-color: var(--yellow); }
.s-rerank  { --glow-color: rgba(167,139,250,0.5); } .s-rerank.done,  .s-rerank.active  { border-color: var(--violet); }
.s-llm     { --glow-color: rgba(6,255,165,0.5);   } .s-llm.done,     .s-llm.active     { border-color: var(--mint); }
.answer-medallion { background: var(--bg-panel); border-radius: 32px; border: 2px solid rgba(6,255,165,0.25); padding: 2rem 2.2rem; margin-top: 1rem; position: relative; }
.answer-medallion::before { content: "💡"; position: absolute; top: -18px; left: 32px; background: var(--mint); width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1rem; }
.answer-label { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; letter-spacing: 0.1em; color: var(--mint); text-transform: uppercase; margin-bottom: 0.8rem; margin-left: 0.2rem; }
.answer-text { color: var(--text-main); font-size: 1.05rem; line-height: 1.6; }
.stat-row { display: flex; gap: 0.9rem; margin-top: 1.3rem; flex-wrap: wrap; }
.stat-pill { background: var(--bg-panel-light); border-radius: 50px; padding: 0.55rem 1.1rem; font-family: 'JetBrains Mono', monospace; font-size: 0.76rem; color: var(--text-dim); border: 1px solid rgba(255,255,255,0.06); }
.stat-pill b { color: var(--text-main); }
.source-bubble { background: var(--bg-panel); border: 1px solid rgba(255,255,255,0.08); border-radius: 28px; padding: 1.3rem 1.6rem; margin-bottom: 0.8rem; display: flex; gap: 1.2rem; align-items: flex-start; }
.source-num { flex-shrink: 0; width: 46px; height: 46px; border-radius: 50%; background: linear-gradient(135deg, var(--violet), var(--coral)); display: flex; align-items: center; justify-content: center; font-weight: 700; color: #fff; font-size: 1rem; }
.source-body { flex: 1; }
.source-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem; }
.source-tag { font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: var(--violet); font-weight: 700; }
.source-score { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: var(--text-dim); background: rgba(167,139,250,0.12); padding: 0.15rem 0.6rem; border-radius: 50px; }
.source-file { font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: var(--turquoise); margin-bottom: 0.3rem; }
.source-snippet { color: var(--text-dim); font-size: 0.87rem; line-height: 1.5; }
.source-keyword-badge { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: var(--yellow); background: rgba(255,230,109,0.1); border: 1px solid rgba(255,230,109,0.3); padding: 0.1rem 0.5rem; border-radius: 50px; margin-left: 0.4rem; }

/* ── VERSION HISTORY SIDEBAR STYLES ── */
.doc-block { background: var(--bg-panel-light); border-radius: 16px; padding: 0.8rem 1rem; margin-top: 0.6rem; border: 1px solid rgba(255,255,255,0.05); }
.doc-block-name { font-weight: 600; color: var(--text-main); font-size: 0.82rem; word-break: break-all; margin-bottom: 0.5rem; }
.version-row { display: flex; align-items: center; gap: 0.5rem; padding: 0.3rem 0; border-bottom: 1px solid rgba(255,255,255,0.04); }
.version-row:last-child { border-bottom: none; }
.version-badge { font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; font-weight: 700; padding: 0.1rem 0.55rem; border-radius: 50px; flex-shrink: 0; }
.v-latest { background: rgba(6,255,165,0.15); color: var(--mint); border: 1px solid rgba(6,255,165,0.3); }
.v-old    { background: rgba(139,141,168,0.12); color: var(--text-dim); border: 1px solid rgba(139,141,168,0.2); }
.version-meta { font-family: 'JetBrains Mono', monospace; font-size: 0.66rem; color: var(--text-dim); flex: 1; }
.version-new-badge { font-family: 'JetBrains Mono', monospace; font-size: 0.62rem; color: var(--yellow); background: rgba(255,230,109,0.12); border: 1px solid rgba(255,230,109,0.3); padding: 0.08rem 0.4rem; border-radius: 50px; }

[data-testid="stFileUploader"] { background: var(--bg-panel); border-radius: 24px; padding: 1rem; border: 2px dashed rgba(78,205,196,0.3); }
[data-testid="stFileUploader"] section { background: transparent; }
.stTextInput input { background: var(--bg-panel) !important; color: var(--text-main) !important; border: 2px solid rgba(255,255,255,0.08) !important; border-radius: 50px !important; font-size: 1rem !important; padding: 0.8rem 1.4rem !important; }
.stTextInput input:focus { border-color: var(--violet) !important; box-shadow: 0 0 0 3px rgba(167,139,250,0.15) !important; }
.stButton button { background: linear-gradient(90deg, var(--coral), var(--violet)) !important; color: white !important; border: none !important; border-radius: 50px !important; font-weight: 600 !important; padding: 0.7rem 1.8rem !important; }
section[data-testid="stSidebar"] { background: var(--bg-panel); }
section[data-testid="stSidebar"] * { color: var(--text-main) !important; }
.file-progress-row { display: flex; align-items: center; gap: 0.7rem; background: var(--bg-panel-light); border-radius: 50px; padding: 0.45rem 1rem; margin-bottom: 0.4rem; font-size: 0.8rem; border: 1px solid rgba(255,255,255,0.05); }
.file-progress-name { flex: 1; color: var(--text-dim); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.file-progress-status { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; }
.status-done { color: var(--mint); } .status-active { color: var(--yellow); } .status-idle { color: var(--text-dim); }
</style>
""", unsafe_allow_html=True)


#  SESSION STATE INIT 
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("BAAI/bge-m3")
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", automodel_args={"torch_dtype": "auto"})
    client = QdrantClient(host="localhost", port=6333)
    return embedder, reranker, client


if "indexed_docs" not in st.session_state:
    st.session_state.indexed_docs = {}
if "next_point_id" not in st.session_state:
    st.session_state.next_point_id = 0


def ensure_collection(client):
    if not client.collection_exists("documents"):
        client.create_collection(
            collection_name="documents",
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )


def render_file_statuses(statuses):
    html = ""
    for name, status in statuses.items():
        short = name[:28] + "…" if len(name) > 30 else name
        if status == "done":
            badge = '<span class="file-progress-status status-done">✓ indexé</span>'
        elif status == "active":
            badge = '<span class="file-progress-status status-active">⏳ en cours…</span>'
        else:
            badge = '<span class="file-progress-status status-idle">— en attente</span>'
        html += f'<div class="file-progress-row"><span class="file-progress-name">📄 {short}</span>{badge}</div>'
    return html


#  SIDEBAR 
with st.sidebar:
    st.markdown("### 📁 Documents sources")

    uploaded_files = st.file_uploader(
        "Glisse tes PDFs ici",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    #  Detect new uploads vs. new VERSIONS of existing files 
    # A file is a "new version" if its name already exists in indexed_docs.
    # A file is "new" if it has never been indexed.
    new_files   = []
    update_files = []
    if uploaded_files:
        st.markdown("**Fichiers sélectionnés :**")
        for f in uploaded_files:
            if f.name in st.session_state.indexed_docs:
                n = len(st.session_state.indexed_docs[f.name]["versions"])
                st.markdown(f"`🔄 {f.name}` → sera indexé comme **v{n+1}**")
                update_files.append(f)
            else:
                st.markdown(f"`🆕 {f.name}`")
                new_files.append(f)

    files_to_index = new_files + update_files

    if files_to_index:
        new_count    = len(new_files)
        update_count = len(update_files)
        parts = []
        if new_count:    parts.append(f"{new_count} nouveau{'x' if new_count > 1 else ''}")
        if update_count: parts.append(f"{update_count} mise{'s' if update_count > 1 else ''} à jour")
        btn_label = f"🔄 Indexer — {' · '.join(parts)}"

        if st.button(btn_label, use_container_width=True):
            embedder, reranker, client = load_models()
            ensure_collection(client)

            st.markdown("**Indexation en cours…**")
            status_slot = st.empty()
            file_statuses = {f.name: "idle" for f in files_to_index}
            status_slot.markdown(render_file_statuses(file_statuses), unsafe_allow_html=True)

            for uploaded in files_to_index:
                file_statuses[uploaded.name] = "active"
                status_slot.markdown(render_file_statuses(file_statuses), unsafe_allow_html=True)

                # Save file (append version number to avoid overwriting) 
                version_num = next_version_number(uploaded.name)
                version_id  = make_version_id(uploaded.name, version_num)
                stem        = Path(uploaded.name).stem
                save_path   = DOCS_DIR / f"{stem}_v{version_num}.pdf"
                with open(save_path, "wb") as f:
                    f.write(uploaded.getbuffer())

                t_start = time.time()

                #  Extract & convert tables 
                pipeline_options = PdfPipelineOptions()
                pipeline_options.do_ocr = False

                converter = DocumentConverter(
                format_options={
                      InputFormat.PDF: PdfFormatOption(
                         pipeline_options=pipeline_options
                      )
                 }
            )
                result      = converter.convert(str(save_path))
                text_plain  = _markdown_tables_to_text(result.document.export_to_markdown())

                #  Semantic chunking
                llama_doc   = Document(text=text_plain)
                embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
                splitter    = SemanticSplitterNodeParser(
                    buffer_size=1,
                    breakpoint_percentile_threshold=95,
                    embed_model=embed_model,
                )
                chunks = splitter.get_nodes_from_documents([llama_doc])

                #add_metadata — spec de ton document de conception 
                # Chaque chunk LlamaIndex reçoit ses métadonnées AVANT
                # l'embedding, exactement comme défini avec ton collaborateur :
                #   chunk.metadata = {
                #       "document_id" : nom du fichier source
                #       "version"     : "v1", "v2", …
                #       "date"        : date d'indexation (YYYY-MM-DD)
                #       "page"        : numéro de page si disponible, sinon chunk index
                #   }
                # Ces métadonnées sont ensuite recopiées dans le payload
                # Qdrant pour permettre le filtrage à la requête.
                def add_metadata(chunk, doc_id: str, version: str, date: str, chunk_idx: int):
                    page = (
                        chunk.metadata.get("page_label") or
                        chunk.metadata.get("page_number") or
                        chunk_idx
                    )
                    chunk.metadata.update({
                        "document_id": doc_id,
                        "version":     version,
                        "date":        date,
                        "page":        page,
                    })
                    return chunk

                index_date = datetime.now().strftime("%Y-%m-%d")
                version_label = f"v{version_num}"
                doc_id = uploaded.name  # identifiant stable = nom du fichier

                chunks = [
                    add_metadata(c, doc_id, version_label, index_date, i)
                    for i, c in enumerate(chunks)
                ]

                #  Construire les textes enrichis depuis chunk.metadata 
                doc_name       = uploaded.name.replace("_", " ").replace(".pdf", "")
                texts_enriched = []
                texts_original = []

                for c in chunks:
                    original = c.text
                    section  = (
                        c.metadata.get("section_summary", "") or
                        c.metadata.get("header_path", "")     or
                        c.metadata.get("title", "")
                    )
                    prefix = (
                        f"Document : {doc_name}\n"
                        f"Version  : {c.metadata['version']}\n"
                        f"Date     : {c.metadata['date']}\n"
                        f"Page     : {c.metadata['page']}"
                    )
                    if section:
                        prefix += f"\nSection  : {section}"
                    texts_enriched.append(f"{prefix}\n\n{original}")
                    texts_original.append(original)

                embeddings = embedder.encode(texts_enriched, show_progress_bar=False)

                start_id = st.session_state.next_point_id
                points = [
                    PointStruct(
                        id=start_id + i,
                        vector=embeddings[i].tolist(),
                        #  Payload Qdrant = miroir de chunk.metadata 
                        # On recopie exactement les 4 champs de la spec plus
                        # les champs techniques nécessaires au filtrage et à
                        # l'affichage dans l'UI.
                        payload={
                            #  spec 
                            "document_id": chunks[i].metadata["document_id"],
                            "version":     chunks[i].metadata["version"],
                            "date":        chunks[i].metadata["date"],
                            "page":        chunks[i].metadata["page"],
                            #  technique 
                            "text":          texts_original[i],
                            "text_enriched": texts_enriched[i],
                            "source":        uploaded.name,
                            "version_id":    version_id,   # clé de filtre Qdrant
                            "version_num":   version_num,
                            "chunk_id":      i,
                        }
                    )
                    for i in range(len(texts_original))
                ]
                client.upsert(collection_name="documents", points=points)

                elapsed = time.time() - t_start
                st.session_state.next_point_id = start_id + len(texts_original)

                #  Update session_state with versioned metadata 
                version_entry = {
                    "version_id":  version_id,
                    "label":       f"v{version_num}",
                    "n_chunks":    len(texts_original),
                    "time":        elapsed,
                    "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "point_start": start_id,
                    "point_end":   start_id + len(texts_original) - 1,
                }

                if uploaded.name not in st.session_state.indexed_docs:
                    st.session_state.indexed_docs[uploaded.name] = {
                        "versions":        [version_entry],
                        "active_versions": [version_id],   # default: query latest
                    }
                else:
                    st.session_state.indexed_docs[uploaded.name]["versions"].append(version_entry)
                    # When a new version arrives, default selection = latest only.
                    # The user can change this via checkboxes below.
                    st.session_state.indexed_docs[uploaded.name]["active_versions"] = [version_id]

                file_statuses[uploaded.name] = "done"
                status_slot.markdown(render_file_statuses(file_statuses), unsafe_allow_html=True)

            time.sleep(0.6)
            st.rerun()

    # VERSION HISTORY DISPLAY 
    if st.session_state.indexed_docs:
        total_chunks = sum(
            v["n_chunks"]
            for doc in st.session_state.indexed_docs.values()
            for v in doc["versions"]
        )
        st.markdown(
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.72rem;'
            f'color:#8B8DA8;margin-top:1rem;">'
            f'BASE DE CONNAISSANCE — {len(st.session_state.indexed_docs)} doc'
            f'{"s" if len(st.session_state.indexed_docs) > 1 else ""} · {total_chunks} chunks</div>',
            unsafe_allow_html=True
        )

        for fname, doc_meta in st.session_state.indexed_docs.items():
            versions = doc_meta["versions"]
            n        = len(versions)
            short    = fname[:34] + "…" if len(fname) > 36 else fname

            # Version history card
            rows_html = ""
            for i, v in enumerate(versions):
                is_latest  = (i == n - 1)
                badge_cls  = "v-latest" if is_latest else "v-old"
                new_marker = '<span class="version-new-badge">NEW</span>' if is_latest and n > 1 else ""
                rows_html += (
                    f'<div class="version-row">'
                    f'  <span class="version-badge {badge_cls}">{v["label"]}</span>'
                    f'  <span class="version-meta">{v["timestamp"]} · {v["n_chunks"]} chunks · {v["time"]:.1f}s</span>'
                    f'  {new_marker}'
                    f'</div>'
                )

            st.markdown(
                f'<div class="doc-block">'
                f'  <div class="doc-block-name">📄 {short}</div>'
                f'  {rows_html}'
                f'</div>',
                unsafe_allow_html=True
            )

            #  Version selector (shown only when >1 version exists) 
            # Each checkbox is independent; the user can mix versions freely.
            # State is stored back into session_state.indexed_docs immediately.
            if n > 1:
                st.markdown(
                    f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.68rem;'
                    f'color:#8B8DA8;margin:0.4rem 0 0.2rem 0.4rem;">VERSIONS À INTERROGER</div>',
                    unsafe_allow_html=True
                )
                new_active = []
                for v in versions:
                    is_currently_active = v["version_id"] in doc_meta["active_versions"]
                    is_latest           = (v == versions[-1])
                    label               = f"{v['label']}  ({v['timestamp']})" + (" ← dernière" if is_latest else "")
                    checked = st.checkbox(
                        label,
                        value=is_currently_active,
                        key=f"ver__{fname}__{v['version_id']}",
                    )
                    if checked:
                        new_active.append(v["version_id"])

                # Guard: never let the user deselect everything
                if not new_active:
                    new_active = [versions[-1]["version_id"]]
                    st.caption("⚠️ Au moins une version doit être sélectionnée — dernière version rétablie.")

                st.session_state.indexed_docs[fname]["active_versions"] = new_active

        st.markdown("<div style='margin-top:0.8rem;'></div>", unsafe_allow_html=True)
        if st.button("🗑️ Vider l'index", use_container_width=True):
            embedder, reranker, client = load_models()
            if client.collection_exists("documents"):
                client.delete_collection("documents")
            st.session_state.indexed_docs  = {}
            st.session_state.next_point_id = 0
            st.rerun()
    else:
        st.markdown(
            '<div style="font-size:0.78rem;color:#8B8DA8;margin-top:0.5rem;">'
            'Aucun document indexé — glisse des PDFs ci-dessus.</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### ⚙️ Configuration active")
    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.8rem;line-height:2;">
    <b>Embeddings</b><br>BAAI/bge-m3<br><br>
    <b>Chunking</b><br>Sémantique (SemanticSplitter)<br><br>
    <b>Reranker</b><br>BAAI/bge-reranker-base<br><br>
    <b>LLM</b><br>mistral (Ollama, local)<br><br>
    <b>Retrieval</b><br>top_k=15 + keyword fallback → top_n=5 × 1500 chars<br><br>
    <b>Versioning</b><br>Multi-version par document · filtre Qdrant
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.78rem;color:#8B8DA8;">'
        'Pipeline 100% local. Aucune donnée n\'est envoyée à un service externe.</div>',
        unsafe_allow_html=True
    )


#  HERO 
st.markdown("""
<div class="lab-hero">
    <div class="brand-row">
        <div class="brand-mark">OM</div>
        <div>
            <div class="brand-name">Omnishore <span style="color:#8B8DA8;font-weight:400;">— Groupe Medtech</span></div>
            <div class="brand-tagline">Architecte de votre transformation digitale</div>
        </div>
    </div>
    <h1 class="lab-title">Pose ta question<br><span>à tes documents</span></h1>
    <p class="lab-sub">Extraction → Embedding → Qdrant → Reranking → Mistral — pipeline RAG 100% local et vérifiable.</p>
</div>""", unsafe_allow_html=True)


#  PIPELINE VISUAL 
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
        state     = states.get(s["key"], "idle")
        t         = timings.get(s["key"], "")
        time_html = f"{t:.2f}s" if isinstance(t, float) else "—"
        html += (
            f'<div class="station {s["class"]} {state}">'
            f'  <span class="station-icon">{s["icon"]}</span>'
            f'  <div class="station-label">{s["label"]}</div>'
            f'  <div class="station-time">{time_html}</div>'
            f'</div>'
        )
        if i < len(STATIONS) - 1:
            html += '<div class="connector"></div>'
    html += '</div>'
    return html

pipeline_slot = st.empty()
pipeline_slot.markdown(render_pipeline({k["key"]: "idle" for k in STATIONS}, {}), unsafe_allow_html=True)


#  INPUT 
col1, col2 = st.columns([5, 1])
with col1:
    query = st.text_input(
        "Question",
        placeholder="Ex : Quel est l'historique de la société ?",
        label_visibility="collapsed"
    )
with col2:
    ask = st.button("Rechercher →", use_container_width=True)

result_slot = st.empty()


#  PIPELINE EXECUTION 
if ask and not st.session_state.indexed_docs:
    st.warning("Indexe d'abord au moins un document via la barre latérale 📁")

elif ask and query:
    embedder, reranker, client = load_models()
    states  = {k["key"]: "idle" for k in STATIONS}
    timings = {}

    states["extract"]  = "done"
    timings["extract"] = 0.0
    pipeline_slot.markdown(render_pipeline(states, timings), unsafe_allow_html=True)

    #  Embedding 
    states["embed"] = "active"
    pipeline_slot.markdown(render_pipeline(states, timings), unsafe_allow_html=True)
    t0 = time.time()
    query_vector = embedder.encode(query).tolist()
    timings["embed"] = time.time() - t0
    states["embed"]  = "done"
    pipeline_slot.markdown(render_pipeline(states, timings), unsafe_allow_html=True)

    #  Build version filter 
    # get_all_active_version_ids() returns the union of all version_ids the user
    # has checked in the sidebar. We pass this to Qdrant as a must-filter so
    # only chunks from selected versions are retrieved — regardless of how many
    # versions of each document exist in the collection.
    active_ids    = get_all_active_version_ids()
    version_filter = Filter(
        must=[
            FieldCondition(
                key="version_id",
                match=MatchAny(any=active_ids),
            )
        ]
    ) if active_ids else None

    #  Qdrant search 
    states["qdrant"] = "active"
    pipeline_slot.markdown(render_pipeline(states, timings), unsafe_allow_html=True)
    t0 = time.time()

    results = client.query_points(
        collection_name="documents",
        query=query_vector,
        limit=15,
        with_payload=True,
        query_filter=version_filter,     # ← version-aware retrieval
    )

    extra_points = keyword_fallback(client, query, results.points, version_filter)
    all_points   = list(results.points) + extra_points

    timings["qdrant"] = time.time() - t0
    states["qdrant"]  = "done"
    pipeline_slot.markdown(render_pipeline(states, timings), unsafe_allow_html=True)

    #  Reranking 
    states["rerank"] = "active"
    pipeline_slot.markdown(render_pipeline(states, timings), unsafe_allow_html=True)
    t0 = time.time()

    pairs       = [(query, r.payload["text"]) for r in all_points]
    raw_scores  = reranker.predict(pairs)
    norm_scores = [sigmoid(float(s)) for s in raw_scores]
    vector_ids  = {p.id for p in results.points}

    ranked = sorted(
        zip(all_points, norm_scores),
        key=lambda x: x[1],
        reverse=True,
    )[:5]

    timings["rerank"] = time.time() - t0
    states["rerank"]  = "done"
    pipeline_slot.markdown(render_pipeline(states, timings), unsafe_allow_html=True)

    #  LLM 
    states["llm"] = "active"
    pipeline_slot.markdown(render_pipeline(states, timings), unsafe_allow_html=True)
    t0 = time.time()

    context = "\n\n".join([
        f"[Source {i+1} — {p.payload.get('source','?')} {p.payload.get('version_id','')}]:\n"
        f"{p.payload['text'][:1500]}"
        for i, (p, _) in enumerate(ranked)
    ])

    prompt = f"""Tu es un assistant utile. Réponds TOUJOURS en français, quelle que soit la langue de la question.
Réponds uniquement à partir des sources ci-dessous.
Si la réponse n'est pas dans les sources, dis "Je n'ai pas suffisamment d'informations."
Cite toujours la source (nom du fichier et version) que tu as utilisée.

{context}

Question : {query}
Réponse :"""

    response         = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
    timings["llm"]   = time.time() - t0
    states["llm"]    = "done"
    pipeline_slot.markdown(render_pipeline(states, timings), unsafe_allow_html=True)

    answer     = response["message"]["content"]
    total_time = sum(timings.values())

    with result_slot.container():
        st.markdown(f"""
        <div class="answer-medallion">
            <div class="answer-label">Réponse</div>
            <div class="answer-text">{answer}</div>
            <div class="stat-row">
                <div class="stat-pill">⏱️ Total <b>{total_time:.1f}s</b></div>
                <div class="stat-pill">📥 Prompt <b>{response.get('prompt_eval_count','—')} tokens</b></div>
                <div class="stat-pill">📤 Généré <b>{response.get('eval_count','—')} tokens</b></div>
                <div class="stat-pill">📚 Docs <b>{len(st.session_state.indexed_docs)}</b></div>
                <div class="stat-pill">🔖 Versions actives <b>{len(active_ids)}</b></div>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### 📚 Sources utilisées")

        for i, (point, score) in enumerate(ranked):
            source_file = point.payload.get("source", "inconnu")
            version_id  = point.payload.get("version_id", "")
            is_keyword  = point.id not in vector_ids
            kw_badge    = '<span class="source-keyword-badge">keyword</span>' if is_keyword else ""
            st.markdown(f"""
            <div class="source-bubble">
                <div class="source-num">{i+1}</div>
                <div class="source-body">
                    <div class="source-head">
                        <span class="source-tag">SOURCE {i+1} · {version_id}{kw_badge}</span>
                        <span class="source-score">rerank {score:.3f}</span>
                    </div>
                    <div class="source-file">📄 {source_file}</div>
                    <div class="source-snippet">{point.payload['text'][:400]}…</div>
                </div>
            </div>""", unsafe_allow_html=True)

elif ask and not query:
    st.warning("Tape une question avant de lancer la recherche 🙂")