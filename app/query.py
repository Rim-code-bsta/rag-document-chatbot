from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
import ollama
import time

print("⚙️  Chargement des modèles...")
embedder = SentenceTransformer("BAAI/bge-m3")
reranker = CrossEncoder("BAAI/bge-reranker-base")  # ← plus léger
client = QdrantClient(host="localhost", port=6333)
print("✅ Prêt !\n")

while True:
    query = input("❓ Votre question (ou 'quit' pour quitter) : ")

    if query.lower() == "quit":
        break

    # Recherche
    t1 = time.time()
    query_vector = embedder.encode(query).tolist()
    results = client.query_points(
        collection_name="documents",
        query=query_vector,
        limit=5  # ← réduit de 10 à 5
    )
    t2 = time.time()
    print(f"⏱️  Qdrant search : {t2-t1:.2f}s")

    # Reranking
    pairs = [(query, r.payload["text"]) for r in results.points]
    scores = reranker.predict(pairs)
    ranked = sorted(
        zip(results.points, scores),
        key=lambda x: x[1],
        reverse=True
    )[:2]  # ← réduit de 3 à 2
    t3 = time.time()
    print(f"⏱️  Reranking : {t3-t2:.2f}s")

    # Prompt — chunks tronqués à 200 chars
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

    # Génération
    print("\n🤖 Génération...")
    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}]
    )
    t4 = time.time()

    answer = response["message"]["content"]

    print(f"⏱️  Mistral : {t4-t3:.2f}s")
    print(f"⏱️  Total : {t4-t1:.2f}s")
    print(f"📊 Prompt tokens : {response.get('prompt_eval_count', 'N/A')}")
    print(f"📊 Tokens générés : {response.get('eval_count', 'N/A')}")

    print("\n=== RÉPONSE ===")
    print(answer)

    print("\n=== SOURCES ===")
    for i, (point, score) in enumerate(ranked):
        print(f"[Source {i+1}] score: {score:.3f} — {point.payload['text'][:150]}")

    print("\n" + "─"*50 + "\n")