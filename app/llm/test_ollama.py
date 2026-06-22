# test_ollama.py — tester Mistral via Ollama

import ollama

print("Test de Mistral...")

response = ollama.chat(
    model="mistral",
    messages=[{"role": "user", "content": "Say hello in one sentence."}]
)

print("✅ Réponse de Mistral :")
print(response["message"]["content"])