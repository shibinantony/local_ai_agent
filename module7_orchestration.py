from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- CONFIGURATION ---
LLM_MODEL = "llama3" # Or phi3, whatever you pulled in Module 4

def test_langchain_primitives():
    print("--- TESTING LANGCHAIN PRIMITIVES ---")
    
    # 1. THE LLM (The Engine)
    print(f"[+] Initializing ChatOllama ({LLM_MODEL})...")
    llm = ChatOllama(model=LLM_MODEL, temperature=0.4)
    
    # 2. THE PROMPT TEMPLATE (The Blueprint)
    # Instead of hardcoding strings, we create reusable, dynamic templates
    template = """
    You are a Senior AI Architect. 
    Explain the concept of {concept} to a junior developer in exactly three sentences.
    """
    prompt = PromptTemplate.from_template(template)
    
    # 3. THE OUTPUT PARSER (The Filter)
    # This ensures we get a clean string back, stripping away system metadata
    parser = StrOutputParser()
    
    # 4. THE CHAIN (LCEL - LangChain Expression Language)
    # The "|" operator is LangChain magic. It pipes data sequentially from left to right:
    # Variables -> Prompt -> LLM -> Clean Text
    chain = prompt | llm | parser
    
    print("[+] Chain assembled. Invoking...\n")
    
    # 5. EXECUTE
    # We pass the dictionary containing our variable into the start of the chain
    response = chain.invoke({"concept": "Vector Databases"})
    
    print(f"Agent:\n{response}")

if __name__ == "__main__":
    test_langchain_primitives()