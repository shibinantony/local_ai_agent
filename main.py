from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from module2_ingest import process_data_pipeline # Importing your logic from Module 2!
import uvicorn

# 1. INITIALIZE THE APP
app = FastAPI(
    title="Local GenAI Agent",
    description="A production-grade local RAG API",
    version="1.0.0"
)

# 2. DEFINE DATA MODELS (The Contract)
# This ensures that if someone sends bad data, the API rejects it automatically.
class IngestRequest(BaseModel):
    filename: str
    chunk_size: int = 200 # Default value if not provided

# 3. DEFINE ENDPOINTS (The Counter)

@app.get("/health")
def health_check():
    """Simple ping to check if server is alive."""
    return {"status": "ok", "service": "local-genai-agent"}

@app.post("/ingest")
def run_ingestion(request: IngestRequest):
    """
    Triggers the ETL pipeline for a specific file.
    """
    try:
        print(f"--- API REQUEST: Ingesting {request.filename} ---")
        
        # Reuse the logic you built in Module 2
        df = process_data_pipeline(request.filename)
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"File {request.filename} not found or empty.")
        
        # In a real app, we would save to ChromaDB here. 
        # For now, we return the stats.
        return {
            "status": "success",
            "file": request.filename,
            "chunks_processed": len(df),
            "preview_first_chunk": df.iloc[0]['text'] if not df.empty else ""
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 4. ENTRY POINT
if __name__ == "__main__":
    # Host 0.0.0.0 allows other local machines to reach it (optional)
    # Reload=True means it auto-restarts when you edit code.
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)