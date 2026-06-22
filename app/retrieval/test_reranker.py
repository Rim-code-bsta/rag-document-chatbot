# test_reranker.py — reranker les résultats avec BGE-Reranker-v2

from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient

# Connexion Qdrant
client = QdrantClient(host="localhost", port=6333)

# Modèle embedding pour la requête
embedder = SentenceTransformer("BAAI/bge-m3")

# Reranker
print("Chargement du reranker...")
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

# Requête
query = "What happens when an employee is sick?"
query_vector = embedder.encode(query).tolist()

# Récupérer 10 résultats depuis Qdrant
results = client.query_points(
    collection_name="documents",
    query=query_vector,
    limit=10
)

print(f"\n✅ {len(results.points)} chunks récupérés depuis Qdrant")

# Reranking
pairs = [(query, r.payload["text"]) for r in results.points]
scores = reranker.predict(pairs)

# Trier par score reranker
ranked = sorted(
    zip(results.points, scores),
    key=lambda x: x[1],
    reverse=True
)

print("\n=== RÉSULTATS APRÈS RERANKING ===")
for i, (point, score) in enumerate(ranked[:3]):
    print(f"\n--- Résultat {i+1} (reranker score: {score:.3f}) ---")
    print(point.payload["text"][:300])