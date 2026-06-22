# test_qdrant.py — stocker les embeddings dans Qdrant

from docling.document_converter import DocumentConverter
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Extraction
converter = DocumentConverter()
result = converter.convert("documents/test.pdf")
text = result.document.export_to_markdown()

# Chunking
llama_doc = Document(text=text)
splitter = SentenceSplitter(chunk_size=512, chunk_overlap=100)
chunks = splitter.get_nodes_from_documents([llama_doc])

# Embeddings
print("Génération des embeddings...")
model = SentenceTransformer("BAAI/bge-m3")
texts = [chunk.text for chunk in chunks]
embeddings = model.encode(texts, show_progress_bar=True)

# Qdrant
client = QdrantClient(host="localhost", port=6333)

# Créer la collection (version moderne)
if client.collection_exists("documents"):
    client.delete_collection("documents")

client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
)
print("✅ Collection créée")

# Insérer les vecteurs
points = [
    PointStruct(
        id=i,
        vector=embeddings[i].tolist(),
        payload={"text": texts[i]}
    )
    for i in range(len(texts))
]

client.upsert(collection_name="documents", points=points)
print(f"✅ {len(points)} vecteurs insérés dans Qdrant")

# Test de recherche
query = "What happens when an employee is sick?"
query_vector = model.encode(query).tolist()

results = client.query_points(
    collection_name="documents",
    query=query_vector,
    limit=3
)

print("\n=== TEST DE RECHERCHE ===")
for i, r in enumerate(results.points):
    print(f"\n--- Résultat {i+1} (score: {r.score:.3f}) ---")
    print(r.payload["text"][:300])