# test_chunking.py — découper le texte en chunks

from docling.document_converter import DocumentConverter
from llama_index.core.node_parser import SentenceSplitter

# Extraction
converter = DocumentConverter()
result = converter.convert("documents/test.pdf")
doc = result.document
text = doc.export_to_markdown()

# Chunking
from llama_index.core import Document
llama_doc = Document(text=text)

splitter = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=100
)

chunks = splitter.get_nodes_from_documents([llama_doc])

print(f"Nombre de chunks : {len(chunks)}\n")
for i, chunk in enumerate(chunks[:3]):  # afficher les 3 premiers
    print(f"--- Chunk {i+1} ---")
    print(chunk.text[:300])
    print()