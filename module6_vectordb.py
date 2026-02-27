import chromadb
from chromadb.utils import embedding_functions

# --- CONFIGURATION ---
# We use a persistent client so our data saves to disk, 
# rather than vanishing when the script stops.
DB_PATH = "./chroma_storage"
COLLECTION_NAME = "local_brain_docs"

def setup_database():
    print(f"--- INITIALIZING CHROMADB ---")
    
    # 1. Create a Persistent Client
    # This will create a folder called 'chroma_storage' in your project directory
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # 2. Define the Embedding Function
    # Chroma is smart; we can tell it to use the exact model you downloaded in Module 5.
    # It will automatically convert text to vectors when we add/query data.
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    # 3. Create or Get Collection
    # A 'collection' is like a table in a relational database.
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn
    )
    
    print(f"[+] Connected to collection: '{collection.name}'")
    print(f"[+] Current document count: {collection.count()}")
    
    return collection

def seed_database(collection):
    """Inserts sample data into the database."""
    print("\n--- SEEDING DATABASE ---")
    
    # Let's add 3 distinct "knowledge" chunks
    documents = [
        "The company VPN address changed to vpn.enterprise.local on January 1st.",
        "To reset your password, contact the IT helpdesk at extension 4455.",
        "The project codename for the new AI architecture is 'LOCAL_BRAIN'."
    ]
    
    # Every document needs a unique ID and optional metadata
    ids = ["doc_1", "doc_2", "doc_3"]
    metadatas = [{"source": "it_memo"}, {"source": "it_memo"}, {"source": "architecture_doc"}]
    
    # Upsert means "Insert, or Update if it already exists"
    collection.upsert(
        documents=documents,
        ids=ids,
        metadatas=metadatas
    )
    
    print(f"[+] Successfully embedded and stored {len(documents)} documents.")
    print(f"[+] New document count: {collection.count()}")

def search_database(collection, query_text):
    """Searches the database using Semantic Similarity."""
    print(f"\n--- QUERYING DATABASE ---")
    print(f"User Question: '{query_text}'")
    
    results = collection.query(
        query_texts=[query_text],
        n_results=1 # We only want the top 1 most relevant result
    )
    
    # Extracting the best match from the results dictionary
    best_match = results['documents'][0][0]
    best_metadata = results['metadatas'][0][0]
    distance = results['distances'][0][0] # Lower distance = closer match
    
    print(f"\n[+] Top Match Found:")
    print(f"    - Text: {best_match}")
    print(f"    - Source: {best_metadata['source']}")
    print(f"    - Distance Score: {distance:.4f}")

if __name__ == "__main__":
    # 1. Connect
    db_collection = setup_database()
    
    # 2. Add Data
    seed_database(db_collection)
    
    # 3. Search Data
    test_query = "What is the new web address for the virtual private network?"
    search_database(db_collection, test_query)