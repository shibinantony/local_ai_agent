from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- CONFIGURATION ---
LLM_MODEL = "llama3"
DB_PATH = "./chroma_storage"
COLLECTION_NAME = "local_brain_docs"

def build_and_run_rag():
    print("--- 1. CONNECTING TO MEMORY (ChromaDB) ---")
    # We use the exact same embedding model to ensure the math matches
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Connect to the database we built in Module 6
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )
    
    # Turn the database into a LangChain "Retriever"
    # k=1 means "Retrieve the top 1 most relevant document chunk"
    retriever = vector_store.as_retriever(search_kwargs={"k": 1})
    print(f"[+] Connected to database. Total docs available: {vector_store._collection.count()}")

    print("\n--- 2. WAKING UP THE BRAIN (Ollama) ---")
    # Temperature 0.1 keeps the AI focused and factual, reducing hallucinations
    llm = ChatOllama(model=LLM_MODEL, temperature=0.1)
    
    print("\n--- 3. ASSEMBLING THE RAG PIPELINE ---")
    # This prompt forces the AI to ONLY use the provided context.
    template = """
    You are 'LOCAL_BRAIN', a secure, locally-hosted AI agent.
    Answer the user's question based ONLY on the following context. 
    If you do not know the answer based on the context, say "I don't have that information in my local memory."
    Do not make things up.

    Context:
    {context}

    Question: {question}

    Answer:
    """
    prompt = PromptTemplate.from_template(template)

    # Helper function to format the retrieved documents into a single string
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # THE LCEL CHAIN (The core of modern LangChain)
    # 1. Takes the user question and passes it to the retriever.
    # 2. Formats the retrieved docs into the 'context' variable.
    # 3. Passes the original question into the 'question' variable.
    # 4. Feeds both to the Prompt -> LLM -> Output Parser.
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("[+] Pipeline assembled successfully.")

    # --- EXECUTION ---
    print("\n--- 4. INITIATING QUERY ---")
    
    # Try asking a question that relies entirely on the custom data we seeded in Module 6
    user_question = "What is the new VPN address for the company?"
    print(f"User: {user_question}")
    
    print("\nProcessing... (Retrieving docs and generating answer)\n")
    response = rag_chain.invoke(user_question)
    
    print(f"LOCAL_BRAIN Agent:\n{response}")

if __name__ == "__main__":
    build_and_run_rag()