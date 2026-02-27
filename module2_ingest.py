import pandas as pd
from pathlib import Path
from typing import List, Dict
import datetime

# --- CONFIGURATION ---
DATA_DIR = Path("data")
CHUNK_SIZE = 200  # Characters per chunk (Simulating token limits)

def load_document(filename: str) -> str:
    """Reads a text file and returns raw string."""
    file_path = DATA_DIR / filename
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"[!] Error: File {filename} not found in {DATA_DIR}")
        return ""

def simple_chunker(text: str, size: int) -> List[str]:
    """
    Splits text into fixed-size chunks.
    (In Module 7, we will upgrade this to 'Semantic Chunking' using LangChain)
    """
    # List comprehension to slice the string
    return [text[i : i + size] for i in range(0, len(text), size)]

def process_data_pipeline(filename: str) -> pd.DataFrame:
    """Orchestrates the ETL process."""
    print(f"--- STARTING ETL PIPELINE FOR: {filename} ---")
    
    # 1. EXTRACT
    raw_text = load_document(filename)
    if not raw_text:
        return pd.DataFrame() # Return empty if failed
    
    print(f"[+] Extracted {len(raw_text)} characters.")

    # 2. TRANSFORM (Chunking)
    chunks = simple_chunker(raw_text, CHUNK_SIZE)
    print(f"[+] Created {len(chunks)} chunks.")

    # 3. LOAD (Structuring)
    # We add metadata (source, timestamp) to every chunk.
    # This is CRITICAL for RAG so you know *where* the answer came from.
    data_payload = []
    for i, chunk in enumerate(chunks):
        record = {
            "chunk_id": i,
            "text": chunk,
            "source": filename,
            "created_at": datetime.datetime.now().isoformat(),
            "char_count": len(chunk)
        }
        data_payload.append(record)

    # Convert to Pandas DataFrame
    df = pd.DataFrame(data_payload)
    return df

if __name__ == "__main__":
    # Run the pipeline
    df_result = process_data_pipeline("sample_mission.txt")
    
    # Validation: Inspect the output
    print("\n--- PIPELINE OUTPUT PREVIEW ---")
    if not df_result.empty:
        print(df_result.head()) # Show first 5 rows
        print(f"\n[+] Dataframe Shape: {df_result.shape} (Rows, Columns)")
        
        # Verify we can export it (simulating a database save)
        # df_result.to_json("debug_storage.json", orient="records")
        # print("[+] Saved debug snapshot to JSON.")
    else:
        print("[!] Pipeline failed.")