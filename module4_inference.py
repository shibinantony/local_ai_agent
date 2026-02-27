import requests
import json
import time

# --- CONFIGURATION ---
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3" # Change to "phi3" if you downloaded that instead

def query_local_llm(prompt: str) -> str:
    """
    Sends a prompt to the local Ollama instance and returns the response.
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False  # For simple scripts, we turn off streaming to get the full text at once
    }
    
    try:
        start_time = time.time()
        
        # 1. SEND REQUEST (The API Call)
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status() # Raise error if status is not 200
        
        # 2. PARSE RESPONSE
        data = response.json()
        actual_response = data.get("response", "")
        
        # Performance Logging
        duration = time.time() - start_time
        print(f"[+] AI Response generated in {duration:.2f} seconds.")
        
        return actual_response

    except requests.exceptions.ConnectionError:
        return "[!] Error: Could not connect to Ollama. Is the app running?"
    except Exception as e:
        return f"[!] Error: {str(e)}"

if __name__ == "__main__":
    print(f"--- TESTING LOCAL LLM ({MODEL_NAME}) ---")
    
    test_question = "Explain the concept of 'Vector Embeddings' in one sentence."
    print(f"User: {test_question}\n")
    
    # Run the Inference
    answer = query_local_llm(test_question)
    
    print(f"Agent: {answer}")