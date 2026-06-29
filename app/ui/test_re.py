from sentence_transformers import CrossEncoder
reranker = CrossEncoder("BAAI/bge-reranker-base")
scores = reranker.predict([
    ("tenue vestimentaire", "AtlasTech adopte une politique vestimentaire business casual"),
    ("tenue vestimentaire", "La visite médicale annuelle est prise en charge"),
])
print(scores)