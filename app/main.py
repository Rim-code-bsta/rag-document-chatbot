# main.py — pipeline RAG complet

from docling.document_converter import DocumentConverter
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import ollama

#  1. EXTRACTION 
print("📄 Extraction du PDF...")
converter = DocumentConverter()
result = converter.convert("documents/test.pdf")
text = result.document.export_to_markdown()

#  2. CHUNKING 
print("✂️  Chunking...")
llama_doc = Document(text=text)
splitter = SentenceSplitter(chunk_size=512, chunk_overlap=100)
chunks = splitter.get_nodes_from_documents([llama_doc])
texts = [chunk.text for chunk in chunks]

#  3. EMBEDDINGS 
print("🔢 Génération des embeddings...")
embedder = SentenceTransformer("BAAI/bge-m3")
embeddings = embedder.encode(texts, show_progress_bar=False)

#  4. QDRANT 
print("💾 Stockage dans Qdrant...")
client = QdrantClient(host="localhost", port=6333)

if client.collection_exists("documents"):
    client.delete_collection("documents")

client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
)

points = [
    PointStruct(
        id=i,
        vector=embeddings[i].tolist(),
        payload={"text": texts[i], "chunk_id": i}
    )
    for i in range(len(texts))
]
client.upsert(collection_name="documents", points=points)

#  5. RECHERCHE + RERANKING 
query = input("\n❓ Votre question : ")

print("\n🔍 Recherche dans Qdrant...")
query_vector = embedder.encode(query).tolist()
results = client.query_points(
    collection_name="documents",
    query=query_vector,
    limit=10
)

print("📊 Reranking...")
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
pairs = [(query, r.payload["text"]) for r in results.points]
scores = reranker.predict(pairs)

ranked = sorted(
    zip(results.points, scores),
    key=lambda x: x[1],
    reverse=True
)[:3]

#  6. PROMPT + LLM
context = "\n\n".join([
    f"[Source {i+1}]:\n{point.payload['text']}"
    for i, (point, score) in enumerate(ranked)
])

prompt = f"""You are a helpful assistant. Answer the question based ONLY on the sources below.
If the answer is not in the sources, say "I don't have enough information."
Always cite which source you used.

{context}

Question: {query}
Answer:"""

print("\n🤖 Génération de la réponse...")
response = ollama.chat(
    model="mistral",
    messages=[{"role": "user", "content": prompt}]
)

print("\n=== RÉPONSE ===")
print(response["message"]["content"])

print("\n=== SOURCES UTILISÉES ===")
for i, (point, score) in enumerate(ranked):
    print(f"\n[Source {i+1}] (score: {score:.3f})")
    print(point.payload["text"][:200])