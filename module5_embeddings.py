from sentence_transformers import SentenceTransformer
import time
import numpy as np

# --- CONFIGURATION ---
# This model is the "standard" for lightweight local RAG.
# It converts any text into a list of 384 numbers.
MODEL_NAME = "all-MiniLM-L6-v2"

def load_embedding_model():
    print(f"--- LOADING EMBEDDING MODEL: {MODEL_NAME} ---")
    start = time.time()
    model = SentenceTransformer(MODEL_NAME)
    print(f"[+] Model loaded in {time.time() - start:.2f} seconds.")
    return model

def generate_embedding(model, text: str):
    """Converts string -> vector"""
    # encode() returns a numpy array
    vector = model.encode(text)
    return vector

if __name__ == "__main__":
    # 1. Initialize the Brain
    embedder = load_embedding_model()
    
    # 2. Define test data
    doc1 = "The quick brown fox jumps over the lazy dog."
    doc2 = "A fast fox leaps over a sleepy canine."
    doc3 = "The Quantum Mechanics of sub-atomic particles."
    
    print("\n--- GENERATING VECTORS ---")
    
    # 3. Convert to Vectors
    vec1 = generate_embedding(embedder, doc1)
    vec2 = generate_embedding(embedder, doc2)
    vec3 = generate_embedding(embedder, doc3)
    
    # 4. INSPECT THE "MAGIC"
    print(f"Original Text: '{doc1}'")
    print(f"Vector Shape: {vec1.shape} dimensions")
    print(f"First 5 numbers: {vec1[:5]}") # Just peeking at the math
    
    # 5. MATH PROOF (Cosine Similarity)
    # We calculate the Dot Product to see how similar they are.
    # Higher score = More similar.
    similarity_1_2 = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    similarity_1_3 = np.dot(vec1, vec3) / (np.linalg.norm(vec1) * np.linalg.norm(vec3))
    
    print(f"\n--- SEMANTIC SIMILARITY CHECK ---")
    print(f"Similarity (Fox vs. Canine): {similarity_1_2:.4f}  <-- Should be HIGH")
    print(f"Similarity (Fox vs. Physics): {similarity_1_3:.4f}  <-- Should be LOW")