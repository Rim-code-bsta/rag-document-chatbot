from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)

results = client.scroll(
    collection_name="documents",
    limit=1000,
    with_payload=True,
    with_vectors=False
)

for p in results[0]:
    source = p.payload.get("source", "")

    if "Doc2" in source:
        text = p.payload.get("text", "")

        if (
            "Historique" in text
            or "2015" in text
            or "2017" in text
            or "2019" in text
            or "2022" in text
            or "2025" in text
        ):
            print("\n====================")
            print(source)
            print(text)
            print("====================")