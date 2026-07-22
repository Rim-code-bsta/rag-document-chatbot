# app/ui/streamlit_app.py — RAG Omnishore · version avec gestion des versions de documents
import json
import uuid
import shutil
from urllib.parse import quote
from datetime import datetime
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
    Filter, FieldCondition, MatchAny, MatchValue,
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

# STATIC FILE SERVING (pour ouvrir le PDF source original au clic)
# Streamlit ne sert des fichiers statiques QUE depuis un dossier "static/"
# situé à côté du script en cours d'exécution (pas depuis un dossier
# arbitraire, et pas depuis le répertoire de lancement). Cela nécessite aussi
# d'activer enableStaticServing = true dans .streamlit/config.toml.
# Voir : https://docs.streamlit.io/develop/concepts/configuration/serving-static-files
STATIC_DOCS_DIR = Path(__file__).resolve().parent / "static" / "documents"
STATIC_DOCS_DIR.mkdir(parents=True, exist_ok=True)


def sync_static_docs():
    """
    Copie dans static/documents/ tout PDF présent dans documents/ mais pas
    encore synchronisé. Appelée à chaque exécution du script (donc aussi
    juste après l'indexation d'un nouveau document, via st.rerun()) — idem-
    potente, ne recopie pas les fichiers déjà présents.
    """
    for pdf_path in DOCS_DIR.glob("*.pdf"):
        target = STATIC_DOCS_DIR / pdf_path.name
        if not target.exists():
            shutil.copy2(pdf_path, target)


sync_static_docs()

# KEYWORD MAP
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


# VERSIONING HELPERS
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
# allowing Qdrant's must filter to restrict retrieval to selected versions only.
#
# IMPORTANT: this dict lives in st.session_state, which is per-session and is
# NOT persisted anywhere. Qdrant itself IS persisted. So on every fresh
# session (app restart, new browser tab, etc.) we rebuild this dict directly
# from what's actually stored in Qdrant — see rebuild_indexed_docs() below.
# This is what fixes the "please re-index" / duplicate-version bug.

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


def rebuild_indexed_docs(client: QdrantClient) -> dict:
    """
    Reconstructs the indexed_docs registry (filename -> versions -> chunk counts,
    etc.) directly from what's actually stored in Qdrant, instead of relying on
    st.session_state, which is wiped every time the Streamlit process restarts
    or a new session starts.

    This is the source-of-truth rebuild: Qdrant is authoritative, session_state
    is just a cache of it for display/filtering convenience.
    """
    if not client.collection_exists("documents"):
        return {}

    points, _ = client.scroll(
        collection_name="documents",
        limit=100_000,
        with_payload=["document_id", "version", "version_id", "date"],
        with_vectors=False,
    )

    if not points:
        return {}

    docs: dict[str, dict] = {}

    for p in points:
        payload = p.payload
        fname = payload.get("document_id") or payload.get("source")
        if not fname:
            continue

        vid = payload.get("version_id", "")
        docs.setdefault(fname, {"versions": {}, "active_versions": []})

        entry = docs[fname]["versions"].setdefault(vid, {
            "version_id": vid,
            "label": payload.get("version", "v1"),
            "n_chunks": 0,
            "time": 0.0,
            "timestamp": payload.get("date", ""),
        })
        entry["n_chunks"] += 1

    # Turn each document's version dict into a sorted list (by version_id,
    # which sorts correctly since it ends in ::v1, ::v2, ...) and default the
    # active selection to the latest version only, matching the behaviour
    # of a freshly indexed document.
    for fname, doc in docs.items():
        versions_list = sorted(
            doc["versions"].values(),
            key=lambda v: v["version_id"],
        )
        doc["versions"] = versions_list
        doc["active_versions"] = [versions_list[-1]["version_id"]]

    return docs


# KEYWORD DETECTION
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
        scroll_filter=version_filter,   # respect version selection
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


# CONTEXT EXPANSION
# For each top-reranked chunk, fetch its immediate previous/next neighbor
# (same document + version) and stitch them together before handing the
# text to the LLM. This directly targets the chunk-splitting problem
# identified earlier: when the semantic chunker cuts an enumerated section
# (e.g. "types de facture") mid-way, the reranker may surface only one half
# — this ensures the other half rides along as extra context even then,
# rather than being omitted from generation entirely.
def expand_with_neighbors(client, ranked_points: list) -> dict:
    """
    Returns {point.id: expanded_text} for each point in ranked_points,
    where expanded_text is [previous_chunk?] + current_chunk + [next_chunk?]
    joined with blank lines. Points without usable document_id/version_id/
    chunk_id metadata (e.g. very old indexed data) fall back to just their
    own text, unchanged.
    """
    expanded = {}

    for point, _ in ranked_points:
        doc_id     = point.payload.get("document_id")
        version_id = point.payload.get("version_id")
        chunk_id   = point.payload.get("chunk_id")
        own_text   = point.payload.get("text", "")

        if doc_id is None or version_id is None or not isinstance(chunk_id, int):
            expanded[point.id] = own_text
            continue

        neighbor_filter = Filter(
            must=[
                FieldCondition(key="document_id", match=MatchValue(value=doc_id)),
                FieldCondition(key="version_id",  match=MatchValue(value=version_id)),
                FieldCondition(key="chunk_id",    match=MatchAny(any=[chunk_id - 1, chunk_id + 1])),
            ]
        )
        neighbors, _ = client.scroll(
            collection_name="documents",
            scroll_filter=neighbor_filter,
            limit=2,
            with_payload=True,
            with_vectors=False,
        )

        prev_text = next((n.payload["text"] for n in neighbors if n.payload.get("chunk_id") == chunk_id - 1), None)
        next_text = next((n.payload["text"] for n in neighbors if n.payload.get("chunk_id") == chunk_id + 1), None)

        parts = [t for t in (prev_text, own_text, next_text) if t]
        expanded[point.id] = "\n\n".join(parts)

    return expanded


# TABLE TO TEXT CONVERTER
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


# SIGMOID
def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


# INLINE CITATION LINKS
# Matches "[Source 2]" or "(Source 2)", case-insensitive. Each match is
# turned into a clickable number linking directly to the ORIGINAL PDF
# document (opened in a new tab, jumping to the right page when possible),
# rather than just scrolling to the retrieved chunk snippet — so the user
# can read the full surrounding context manually if they want to.
# The model is instructed to use the bracket form, but small local models
# don't always follow formatting instructions exactly (same lesson as the
# unanswered-question detection), so both bracket and parenthesis forms are
# accepted.
import re

_CITATION_PATTERN     = re.compile(r'[\[\(]\s*[Ss]ource\s*(\d+)\s*[\]\)]')
_VERSION_SUFFIX_RE    = re.compile(r'_v\d+$', re.IGNORECASE)


def _normalize_stem(stem: str) -> str:
    """Strips a trailing _v<N> suffix, if present, for fuzzy comparison."""
    return _VERSION_SUFFIX_RE.sub('', stem)


def resolve_static_pdf_filename(document_id: str, version_label: str) -> str | None:
    """
    Finds the actual filename of a document among the synced static PDFs.
    Rather than blindly reconstructing "{stem}_{version}.pdf" (which breaks
    the moment the real filename doesn't follow that exact pattern — e.g. a
    document whose ORIGINAL uploaded name already contained "_v1" before the
    versioning system existed, or a legacy file saved without any version
    suffix at all), this tries progressively looser matches:
      1. exact "{stem}_{version_label}.pdf"  (current naming convention)
      2. exact "{stem}.pdf"                  (legacy, no version suffix)
      3. any static PDF whose name, once a trailing "_vN" is stripped from
         BOTH sides, matches the target — this is what recovers cases like
         document_id stem "Report_v1" matching an actual file "Report.pdf".
    Returns the matching filename (not the full path), or None if nothing
    is found — callers must handle that case (no link, rather than a
    broken one).
    """
    stem = Path(document_id).stem

    for candidate in (f"{stem}_{version_label}.pdf", f"{stem}.pdf"):
        if (STATIC_DOCS_DIR / candidate).exists():
            return candidate

    normalized_target = _normalize_stem(stem).lower()
    for pdf_path in STATIC_DOCS_DIR.glob("*.pdf"):
        if _normalize_stem(pdf_path.stem).lower() == normalized_target:
            return pdf_path.name

    return None


def build_source_href(document_id: str, version_label: str, page) -> str | None:
    """
    Builds the URL to the original PDF, served by Streamlit's static file
    feature from static/documents/ (see STATIC_DOCS_DIR / sync_static_docs
    near the top of the file). Appends a #page=N fragment when a numeric
    page is available, which Chrome/Edge/Firefox's built-in PDF viewer will
    honor by opening directly at that page. Returns None if no matching
    file can be found — callers should fall back to plain (non-clickable)
    text in that case rather than link to a 404.
    """
    filename = resolve_static_pdf_filename(document_id, version_label)
    if filename is None:
        return None
    href = f"app/static/documents/{quote(filename)}"
    if isinstance(page, (int, float)) and not isinstance(page, bool):
        href += f"#page={int(page)}"
    return href


def linkify_citations(answer: str, source_hrefs: list) -> str:
    def _replace(m):
        n = int(m.group(1))
        if 1 <= n <= len(source_hrefs) and source_hrefs[n - 1]:
            href = source_hrefs[n - 1]
            return f'<a href="{href}" target="_blank" class="citation-link" title="Ouvrir le document source">{n}</a>'
        # Out-of-range source number, or no matching file found for this
        # source (see build_source_href) — leave the original text
        # untouched rather than link to a 404.
        return m.group(0)

    return _CITATION_PATTERN.sub(_replace, answer)


# CSS
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
.source-file-link { display: inline-block; font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: var(--turquoise) !important; margin-bottom: 0.3rem; text-decoration: none; }
.source-file-link:hover { text-decoration: underline; }
.source-snippet { color: var(--text-dim); font-size: 0.87rem; line-height: 1.5; }
.source-keyword-badge { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: var(--yellow); background: rgba(255,230,109,0.1); border: 1px solid rgba(255,230,109,0.3); padding: 0.1rem 0.5rem; border-radius: 50px; margin-left: 0.4rem; }

/* CLICKABLE INLINE CITATIONS */
html { scroll-behavior: smooth; }
.citation-link {
    display: inline-flex; align-items: center; justify-content: center;
    width: 20px; height: 20px; margin: 0 2px; vertical-align: middle;
    background: linear-gradient(135deg, var(--violet), var(--coral));
    color: #fff !important; font-family: 'JetBrains Mono', monospace;
    font-weight: 700; font-size: 0.68rem; border-radius: 50%;
    text-decoration: none; transition: transform 0.15s ease;
}
.citation-link:hover { transform: scale(1.25); text-decoration: none; }
.source-bubble { scroll-margin-top: 90px; transition: box-shadow 0.3s ease, border-color 0.3s ease; }
.source-bubble:target {
    border-color: var(--mint);
    box-shadow: 0 0 0 3px rgba(6,255,165,0.3);
    animation: source-flash 1.6s ease-out;
}
@keyframes source-flash {
    0%   { background: rgba(6,255,165,0.18); }
    100% { background: var(--bg-panel); }
}

/* VERSION HISTORY SIDEBAR STYLES */
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

/* ===========================
   CHAT INPUT
=========================== */

/* Remove the white footer */
[data-testid="stBottom"]{
    background: transparent !important;
}

[data-testid="stBottomBlockContainer"]{
    background: transparent !important;
    padding-top: .5rem !important;
    padding-bottom: 1.2rem !important;
}

/* Remove any white wrapper around the bottom area */
.stBottom{
    background: transparent !important;
}

.stBottom > div{
    background: transparent !important;
}

/* Chat input container */
[data-testid="stChatInput"]{
    background: var(--bg-panel) !important;
    border: 2px solid rgba(255,255,255,.08) !important;
    border-radius: 999px !important;
    overflow: hidden !important;
    box-shadow: none !important;
}

/* Internal wrappers */
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] > div > div,
[data-testid="stChatInput"] form,
[data-testid="stChatInput"] form > div{
    background: transparent !important;
    border: none !important;
}

/* Textarea */
[data-testid="stChatInput"] textarea{
    background: var(--bg-panel) !important;
    color: var(--text-main) !important;
    caret-color: var(--mint) !important;
    border: none !important;
    box-shadow: none !important;
    resize: none !important;
}

/* Placeholder */
[data-testid="stChatInput"] textarea::placeholder{
    color: var(--text-dim) !important;
}

/* Focus */
[data-testid="stChatInput"]:focus-within{
    border-color: var(--violet) !important;
    box-shadow: 0 0 0 3px rgba(167,139,250,.18) !important;
}

/* Send button */
[data-testid="stChatInputSubmitButton"]{
    background: linear-gradient(90deg,var(--coral),var(--violet)) !important;
    border-radius: 16px !important;
}

/* Main page background */
[data-testid="stAppViewContainer"]{
    background: var(--bg-deep) !important;
}

[data-testid="stMain"]{
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)


# SESSION STATE INIT

@st.cache_resource
def get_qdrant_client() -> QdrantClient:
    """
    Cached separately from the embedder/reranker so that rebuilding the
    indexed_docs registry on startup is cheap (no need to load the heavy
    embedding/reranking models just to check what's already in Qdrant).
    """
    return QdrantClient(host="localhost", port=6333)


@st.cache_resource
def load_models():
    embedder = SentenceTransformer("BAAI/bge-m3")
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", automodel_args={"torch_dtype": "auto"})
    client = get_qdrant_client()
    return embedder, reranker, client


if "indexed_docs" not in st.session_state:
    # Rebuild from Qdrant (the real source of truth) instead of starting empty.
    # This is what fixes "please index again" + duplicate-version bugs on
    # every new session / app restart: Qdrant already has the data, we just
    # weren't reading it back into session_state before.
    _client = get_qdrant_client()
    st.session_state.indexed_docs = rebuild_indexed_docs(_client)

# ============================
# CHAT HISTORY
# ============================

CHAT_DIR = Path("chat_history")
CHAT_DIR.mkdir(exist_ok=True)

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# ============================
# FEEDBACK & UNANSWERED-QUESTION LOGGING (axes 6 & 9 du doc original)
# ============================
# Two append-only JSONL logs, kept separate from chat_history/*.json so they
# can be read and aggregated independently for the report (e.g. "% negative
# feedback", "list of gaps in the documentation").
FEEDBACK_LOG    = Path("feedback_log.jsonl")
UNANSWERED_LOG  = Path("unanswered_questions.jsonl")

# Match (case-insensitively) against a set of common "no answer" phrasings,
# not just the exact wording requested in the prompt. Small local models like
# Mistral 7B don't reliably stick to an exact instructed phrase — they
# paraphrase ("n'est pas explicitement mentionné...", "il faudrait consulter
# d'autres sources", etc.) — so a single hardcoded string misses most real
# cases. This list is a heuristic, not exhaustive: extend it whenever you
# spot a new "no answer" phrasing slipping through in feedback_stats.py.
NO_INFO_PHRASES = [
    "je n'ai pas suffisamment d'informations",
    "n'est pas explicitement mentionné",
    "n'est pas mentionné dans les sources",
    "ne sont pas mentionnés dans les sources",
    "les sources ne mentionnent pas",
    "les sources fournies ne mentionnent pas",
    "pas mentionné dans les documents",
    "aucune information",
    "je ne trouve pas",
    "je n'ai pas trouvé",
    "consulter d'autres sources",
    "n'est pas disponible dans les sources",
]


def _append_jsonl(path: Path, entry: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def is_unanswered(answer: str) -> bool:
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in NO_INFO_PHRASES)


def log_unanswered(question: str, answer: str, active_ids: list, chat_id: str):
    _append_jsonl(UNANSWERED_LOG, {
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "chat_id":    chat_id,
        "question":   question,
        "answer":     answer,
        "active_versions": active_ids,
    })


def handle_feedback(message_id: str, question: str, answer: str, sources: list, feedback_value: str):
    """
    Button on_click callback (like/dislike). Using on_click rather than a
    plain `if st.button(...)` block guarantees this runs at the moment of the
    click, regardless of whether the surrounding UI block re-renders after
    the resulting Streamlit rerun (it won't, since this lives inside the
    "answer just generated" block, which only executes on the turn a
    question is asked).
    """
    _append_jsonl(FEEDBACK_LOG, {
        "id":        message_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "chat_id":   st.session_state.current_chat_id,
        "question":  question,
        "answer":    answer,
        "feedback":  feedback_value,  # "like" or "dislike"
        "sources":   sources,
    })
    # Reflect the feedback back into the saved conversation too, so it shows
    # up if the conversation is reopened later from the sidebar.
    for m in st.session_state.messages:
        if m.get("id") == message_id:
            m["feedback"] = feedback_value
            break
    save_chat()
    st.toast(
        "Merci pour votre retour ! 🙏" if feedback_value == "like"
        else "Merci, c'est noté — on va améliorer ça. 🛠️"
    )


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

# ============================
# CHAT FUNCTIONS
# ============================

def chat_path(chat_id):
    return CHAT_DIR / f"{chat_id}.json"


def list_chats():
    chats = []
    for file in sorted(
        CHAT_DIR.glob("*.json"),
        reverse=True,
        key=lambda x: x.stat().st_mtime,
    ):
        with open(file, encoding="utf-8") as f:
            data = json.load(f)
        chats.append({"id": file.stem, "title": data["title"]})
    return chats


def save_chat():

    path = chat_path(st.session_state.current_chat_id)

    title = "Nouvelle conversation"

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            title = msg["content"][:40]
            break

    data = {
        "id": st.session_state.current_chat_id,
        "title": title,
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "messages": st.session_state.messages,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    st.session_state.chat_list_cache = list_chats()


def load_chat(chat_id):

    path = chat_path(chat_id)

    if not path.exists():
        return

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    st.session_state.current_chat_id = chat_id
    st.session_state.messages = data["messages"]


def new_chat():

    st.session_state.current_chat_id = str(uuid.uuid4())
    st.session_state.messages = []


def delete_chat(chat_id):

    path = chat_path(chat_id)

    if path.exists():
        path.unlink()

    st.session_state.chat_list_cache = list_chats()

    if chat_id == st.session_state.current_chat_id:
        new_chat()


if "chat_list_cache" not in st.session_state:
    st.session_state.chat_list_cache = list_chats()


# SIDEBAR
with st.sidebar:

    # ============================
    # CONVERSATIONS
    # ============================

    st.markdown("## 💬 Conversations")

    if st.button("➕ Nouvelle conversation", use_container_width=True):
        save_chat()
        new_chat()
        st.rerun()

    for chat in st.session_state.chat_list_cache:

        is_active = chat["id"] == st.session_state.current_chat_id
        label = ("🟢 " if is_active else "") + chat["title"]

        if st.button(label, key=f"open_{chat['id']}", use_container_width=True):
            load_chat(chat["id"])
            st.rerun()

    if st.session_state.chat_list_cache:
        with st.expander("🗑️ Supprimer une conversation"):
            options = {c["title"]: c["id"] for c in st.session_state.chat_list_cache}
            to_delete = st.selectbox("Choisir", list(options.keys()), key="del_select")
            if st.button("Confirmer la suppression", key="confirm_del"):
                delete_chat(options[to_delete])
                st.rerun()

    st.divider()

    # ============================
    # DOCUMENTS
    # ============================
    st.markdown("### 📁 Documents sources")

    uploaded_files = st.file_uploader(
        "Glisse tes PDFs ici",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    # Detect new uploads vs new VERSIONS of existing files
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

                # Extract and convert tables
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

                # Semantic chunking
                llama_doc   = Document(text=text_plain)
                embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
                splitter    = SemanticSplitterNodeParser(
                    buffer_size=1,
                    breakpoint_percentile_threshold=95,
                    embed_model=embed_model,
                )
                chunks = splitter.get_nodes_from_documents([llama_doc])

                # add_metadata, spec de ton document de conception
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

                # Construire les textes enrichis depuis chunk.metadata
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

                # Use UUIDs for point IDs instead of a manually-incremented
                # counter. The counter used to live only in st.session_state,
                # which resets on every new session — causing IDs to restart
                # at 0 and silently overwrite previously indexed vectors in
                # Qdrant. UUIDs remove the need for any cross-session counter.
                points = [
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embeddings[i].tolist(),
                        # Payload Qdrant = miroir de chunk.metadata
                        # On recopie exactement les 4 champs de la spec plus
                        # les champs techniques nécessaires au filtrage et à
                        # l'affichage dans l'UI.
                        payload={
                            # spec
                            "document_id": chunks[i].metadata["document_id"],
                            "version":     chunks[i].metadata["version"],
                            "date":        chunks[i].metadata["date"],
                            "page":        chunks[i].metadata["page"],
                            # technique
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

                # Update session_state with versioned metadata
                version_entry = {
                    "version_id":  version_id,
                    "label":       f"v{version_num}",
                    "n_chunks":    len(texts_original),
                    "time":        elapsed,
                    "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M"),
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

            # Version selector (shown only when more than one version exists)
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
            client = get_qdrant_client()
            if client.collection_exists("documents"):
                client.delete_collection("documents")
            st.session_state.indexed_docs = {}
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


# HERO
st.markdown("""
<div class="lab-hero">
    <div class="brand-row">
        <div class="brand-mark">OM</div>
        <div>
            <div class="brand-name">Omnishore AI</div>
            <div class="brand-tagline">Architecte de votre transformation digitale</div>
        </div>
    </div>
</div>""", unsafe_allow_html=True)


# PIPELINE VISUAL
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


# ============================
# CONVERSATION
# ============================

for msg in st.session_state.messages:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("feedback"):
            st.caption(
                "👍 Marqué comme utile" if msg["feedback"] == "like"
                else "👎 Marqué comme pas utile"
            )

query = st.chat_input("Ex : Quel est l'historique de la société ?")
ask = query is not None

result_slot = st.empty()


# PIPELINE EXECUTION
if ask and not st.session_state.indexed_docs:
    st.warning("Indexe d'abord au moins un document via la barre latérale 📁")

elif ask and query:
    st.session_state.messages.append(
      {
        "role": "user",
        "content": query,
      }
    )

    save_chat()
    embedder, reranker, client = load_models()
    states  = {k["key"]: "idle" for k in STATIONS}
    timings = {}

    states["extract"]  = "done"
    timings["extract"] = 0.0
    pipeline_slot.markdown(render_pipeline(states, timings), unsafe_allow_html=True)

    # Embedding
    states["embed"] = "active"
    pipeline_slot.markdown(render_pipeline(states, timings), unsafe_allow_html=True)
    t0 = time.time()
    query_vector = embedder.encode(query).tolist()
    timings["embed"] = time.time() - t0
    states["embed"]  = "done"
    pipeline_slot.markdown(render_pipeline(states, timings), unsafe_allow_html=True)

    # Build version filter
    # get_all_active_version_ids() returns the union of all version_ids the user
    # has checked in the sidebar. We pass this to Qdrant as a must filter so
    # only chunks from selected versions are retrieved, regardless of how many
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

    # Qdrant search
    states["qdrant"] = "active"
    pipeline_slot.markdown(render_pipeline(states, timings), unsafe_allow_html=True)
    t0 = time.time()

    results = client.query_points(
        collection_name="documents",
        query=query_vector,
        limit=15,
        with_payload=True,
        query_filter=version_filter,     # version-aware retrieval
    )

    extra_points = keyword_fallback(client, query, results.points, version_filter)
    all_points   = list(results.points) + extra_points

    timings["qdrant"] = time.time() - t0
    states["qdrant"]  = "done"
    pipeline_slot.markdown(render_pipeline(states, timings), unsafe_allow_html=True)

    # Reranking
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

    # Context Expansion: fetch the previous/next chunk for each of the top 5
    # selected chunks, so a section split awkwardly by the semantic chunker
    # (see CTX.1 in the improvement-axes doc) still reaches the LLM whole.
    expanded_by_id = expand_with_neighbors(client, ranked)

    timings["rerank"] = time.time() - t0
    states["rerank"]  = "done"
    pipeline_slot.markdown(render_pipeline(states, timings), unsafe_allow_html=True)

    # LLM
    states["llm"] = "active"
    pipeline_slot.markdown(render_pipeline(states, timings), unsafe_allow_html=True)
    t0 = time.time()

    # Keep the rerank order for the "SOURCE N" numbering shown to the user
    # (and used for citations), but build the context passed to the LLM in
    # DOCUMENT ORDER (same source file + page/chunk_id ascending) whenever
    # chunks come from the same document. This matters because the semantic
    # chunker sometimes splits an enumerated section (e.g. "2.1 / 2.2 / 2.3 /
    # 2.4 Types de Factures") across two separate chunks. If those chunks are
    # then fed to the LLM in rerank-score order instead of document order,
    # the tail of one chunk (e.g. the end of item 2.3 + all of item 2.4) can
    # read like a disconnected footnote instead of a continuation of the
    # list, and the model may miss it as a distinct enumerated item.
    def _sort_key(pair):
        p, _ = pair
        page = p.payload.get("page", 0)
        page = page if isinstance(page, (int, float)) else 0
        return (p.payload.get("source", ""), page, p.payload.get("chunk_id", 0))

    ranked_for_context = sorted(ranked, key=_sort_key)
    source_numbers = {id(p): i + 1 for i, (p, _) in enumerate(ranked)}

    # Truncation limit raised from 1500 to 2500 chars per source, since each
    # source now typically includes its neighboring chunks too (Context
    # Expansion). This does increase prompt size / generation time somewhat
    # — a deliberate trade-off given the hardware constraints already
    # documented, favoring completeness over raw speed.
    context = "\n\n".join([
        f"[Source {source_numbers[id(p)]} — {p.payload.get('source','?')} {p.payload.get('version_id','')}]:\n"
        f"{expanded_by_id.get(p.id, p.payload['text'])[:2500]}"
        for p, _ in ranked_for_context
    ])

    prompt = f"""Tu es un assistant utile. Réponds TOUJOURS en français, quelle que soit la langue de la question.
Réponds uniquement à partir des sources ci-dessous.
Si la réponse n'est pas dans les sources, dis "Je n'ai pas suffisamment d'informations."

Citations obligatoires : après CHAQUE affirmation tirée d'une source, insère immédiatement une citation
au format [Source N], où N est exactement le numéro indiqué entre crochets au début de la source
correspondante ci-dessous (par exemple [Source 1] ou [Source 3]). N'utilise jamais le nom du fichier
dans le texte de la réponse — uniquement ce format [Source N]. Si une affirmation s'appuie sur plusieurs
sources, cite-les toutes, par exemple [Source 1][Source 2].

Attention : les sources peuvent contenir des listes énumérées ou numérotées (ex. types, catégories, étapes,
sections 2.1, 2.2, 2.3...). Certaines de ces listes peuvent être coupées entre deux sources différentes.
Avant de répondre, identifie TOUS les éléments numérotés ou énumérés pertinents présents dans l'ensemble
des sources, même s'ils apparaissent en fin d'une source ou semblent être une continuation d'un paragraphe
précédent. Ne t'arrête pas au premier groupe d'éléments trouvé : vérifie chaque source jusqu'au bout.

{context}

Question : {query}
Réponse :"""

    response         = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
    timings["llm"]   = time.time() - t0
    states["llm"]    = "done"
    pipeline_slot.markdown(render_pipeline(states, timings), unsafe_allow_html=True)

    answer     = response["message"]["content"]
    total_time = sum(timings.values())

    # Unique id for this assistant turn, used to key the like/dislike buttons
    # and to re-attach feedback to the right message afterwards.
    message_id = str(uuid.uuid4())

    # Built once here and reused both for the LLM-facing citations and for
    # the persisted message / feedback log, instead of rebuilding it twice.
    sources_payload = [
        {
            "file":    point.payload.get("source", "inconnu"),
            "version": point.payload.get("version_id", ""),
            "score":   float(score),
        }
        for point, score in ranked
    ]

    # Automatic logging of unanswered questions (axe 9) — this runs
    # regardless of whether the user gives explicit like/dislike feedback,
    # so it captures every documentation gap the chatbot hits.
    if is_unanswered(answer):
        log_unanswered(query, answer, active_ids, st.session_state.current_chat_id)

    with result_slot.container():
        # One real PDF link per ranked source, built from document_id +
        # version + page (see build_source_href above). Used both for the
        # inline [Source N] citations in the answer, and for the "open
        # document" link on each source card below.
        source_hrefs = [
            build_source_href(
                p.payload.get("document_id", p.payload.get("source", "")),
                p.payload.get("version", "v1"),
                p.payload.get("page"),
            )
            for p, _ in ranked
        ]
        answer_html = linkify_citations(answer, source_hrefs)
        st.markdown(f"""
        <div class="answer-medallion">
            <div class="answer-label">Réponse</div>
            <div class="answer-text">{answer_html}</div>
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
            href        = source_hrefs[i]
            if href:
                file_line = f'<a href="{href}" target="_blank" class="source-file-link">📄 {source_file} — ouvrir le document ↗</a>'
            else:
                # No matching PDF found in static/documents/ for this
                # source (see resolve_static_pdf_filename) — show the plain
                # file name instead of a link that would 404.
                file_line = f'<div class="source-file">📄 {source_file}</div>'
            st.markdown(f"""
            <div class="source-bubble" id="source-{i+1}">
                <div class="source-num">{i+1}</div>
                <div class="source-body">
                    <div class="source-head">
                        <span class="source-tag">SOURCE {i+1} · {version_id}{kw_badge}</span>
                        <span class="source-score">rerank {score:.3f}</span>
                    </div>
                    {file_line}
                    <div class="source-snippet">{point.payload['text'][:400]}…</div>
                </div>
            </div>""", unsafe_allow_html=True)

        # =====================================
        # FEEDBACK (axe 6 — boutons Like / Dislike)
        # =====================================
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Cette réponse vous a-t-elle été utile ?**")
        fb_col1, fb_col2, fb_col3 = st.columns([1, 1, 6])
        with fb_col1:
            st.button(
                "👍 Utile", key=f"like_{message_id}", use_container_width=True,
                on_click=handle_feedback,
                args=(message_id, query, answer, sources_payload, "like"),
            )
        with fb_col2:
            st.button(
                "👎 Pas utile", key=f"dislike_{message_id}", use_container_width=True,
                on_click=handle_feedback,
                args=(message_id, query, answer, sources_payload, "dislike"),
            )

        # =====================================
        # SAVE THE CONVERSATION
        # =====================================

        st.session_state.messages.append(
           {
                 "id": message_id,
                 "role": "assistant",
                 "content": answer,
                 "sources": sources_payload,
                 "feedback": None,
                 "timestamp": datetime.now().strftime("%H:%M"),
                 "response_time": total_time,
           }
        )

        save_chat()