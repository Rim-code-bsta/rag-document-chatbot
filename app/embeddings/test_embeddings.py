# test_embeddings.py — générer des embeddings avec BGE-M3

from docling.document_converter import DocumentConverter
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document
from sentence_transformers import SentenceTransformer

# Extraction
converter = DocumentConverter()
result = converter.convert("documents/test.pdf")
text = result.document.export_to_markdown()

# Chunking
llama_doc = Document(text=text)
splitter = SentenceSplitter(chunk_size=512, chunk_overlap=100)
chunks = splitter.get_nodes_from_documents([llama_doc])

# Embeddings
print("Chargement du modèle BGE-M3...")
model = SentenceTransformer("BAAI/bge-m3")

texts = [chunk.text for chunk in chunks]
embeddings = model.encode(texts, show_progress_bar=True)

print(f"\n✅ {len(embeddings)} embeddings générés")
print(f"Dimension de chaque vecteur : {len(embeddings[0])}")
print(f"\nAperçu du premier vecteur (5 premières valeurs) :")
print(embeddings[0][:5])