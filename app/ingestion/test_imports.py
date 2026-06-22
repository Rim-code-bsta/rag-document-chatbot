# test_imports.py — ici on ne fait que vérifier que tout s'importe correctement

print("Test des imports...")

try:
    from docling.document_converter import DocumentConverter
    print("✅ Docling OK")
except ImportError as e:
    print(f"❌ Docling manquant : {e}")

try:
    from sentence_transformers import SentenceTransformer
    print("✅ SentenceTransformers OK")
except ImportError as e:
    print(f"❌ SentenceTransformers manquant : {e}")

try:
    from qdrant_client import QdrantClient
    print("✅ Qdrant client OK")
except ImportError as e:
    print(f"❌ Qdrant client manquant : {e}")

try:
    import streamlit
    print("✅ Streamlit OK")
except ImportError as e:
    print(f"❌ Streamlit manquant : {e}")

try:
    import ollama
    print("✅ Ollama OK")
except ImportError as e:
    print(f"❌ Ollama manquant : {e}")

print("\nDone !")