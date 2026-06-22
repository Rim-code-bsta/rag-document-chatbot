
# ingestion.py — à lancer UNE SEULE FOIS pour indexer les documents

from docling.document_converter import DocumentConverter
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

print("📄 Extraction du PDF...")
converter = DocumentConverter()
result = converter.convert("documents/test.pdf")
text = result.document.export_to_markdown()

print("✂️  Chunking...")
llama_doc = Document(text=text)
splitter = SentenceSplitter(chunk_size=512, chunk_overlap=100)
chunks = splitter.get_nodes_from_documents([llama_doc])
texts = [chunk.text for chunk in chunks]
print(f"   → {len(texts)} chunks créés")

print("🔢 Génération des embeddings...")
embedder = SentenceTransformer("BAAI/bge-m3")
embeddings = embedder.encode(texts, show_progress_bar=False)

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

print(f"\n✅ Ingestion terminée — {len(texts)} chunks indexés dans Qdrant")
print("Tu peux maintenant lancer query.py")