# test_docling.py — extraire le texte d'un PDF avec Docling

from docling.document_converter import DocumentConverter

converter = DocumentConverter()


result = converter.convert("documents/test.pdf")
doc = result.document

print("=== TEXTE EXTRAIT ===\n")
print(doc.export_to_markdown())