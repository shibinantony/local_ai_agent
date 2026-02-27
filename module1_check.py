import sys
import pandas as pd
import chromadb
from datetime import datetime

# --- PRODUCTION NOTE: LOGGING ---
# In a real app, we would use the 'logging' library. 
# For this Module 1 check, print statements are acceptable.

def run_diagnostics():
    print(f"--- AI Agent Environment Check ---")
    print(f"Timestamp: {datetime.now()}")
    print(f"Python Version: {sys.version.split()[0]}")
    
    # 1. TEST LISTS (The Batch)
    # Simulating a batch of document titles to ingest
    doc_batch = ["invoice_2024.pdf", "hr_policy.docx", "meeting_notes.txt"]
    print(f"\n[+] List Test: Loaded {len(doc_batch)} dummy documents.")
    
    # 2. TEST DICTIONARIES (The Payload)
    # Simulating an LLM message payload
    llm_payload = {
        "model": "llama-3-local",
        "temperature": 0.7,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Summarize this invoice."}
        ]
    }
    print(f"[+] Dictionary Test: Configured payload for model '{llm_payload['model']}'.")

    # 3. TEST LIBRARIES
    try:
        # distinct check for pandas
        df = pd.DataFrame({"status": ["active"], "latency_ms": [12]})
        print(f"[+] Pandas Test: Library imported successfully.")
        
        # distinct check for ChromaDB
        chroma_client = chromadb.Client()
        print(f"[+] ChromaDB Test: Vector DB client initialized in-memory.")
        
        print(f"\n>>> SYSTEM READY. PROCEED TO MODULE 2. <<<")
        
    except Exception as e:
        print(f"\n[!] CRITICAL ERROR: {e}")
        print("Please check your pip installation.")

if __name__ == "__main__":
    run_diagnostics()