import streamlit as st
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

# 1. UI PAGE SETUP
st.set_page_config(page_title="LOCAL_BRAIN Agent", page_icon="🧠", layout="centered")
st.title("🧠 LOCAL_BRAIN")
st.caption("Secure, Local RAG Agent running on CPU/GPU")

# 2. CACHE THE HEAVY LIFTING
# @st.cache_resource ensures the DB and LLM only load ONCE when the app starts.
@st.cache_resource
def load_rag_pipeline():
    # Embeddings & Memory
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 1})
    
    # Brain
    llm = ChatOllama(model=LLM_MODEL, temperature=0.1)
    
    # Prompt Blueprint
    template = """
    You are 'LOCAL_BRAIN', a secure, locally-hosted AI agent.
    Answer the user's question based ONLY on the following context. 
    If you do not know the answer based on the context, say "I don't have that information in my local memory."
    
    Context:
    {context}

    Question: {question}

    Answer:
    """
    prompt = PromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    # Assemble Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# Initialize the pipeline
with st.spinner("Initializing Local Brain..."):
    chain = load_rag_pipeline()

# 3. SESSION STATE (Chat History)
# This keeps track of the conversation so it doesn't disappear when you hit enter
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. CHAT INTERFACE
# st.chat_input creates the text box at the bottom of the screen
if user_query := st.chat_input("Ask a question about your documents..."):
    
    # Display user query
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate and display AI response
    with st.chat_message("assistant"):
        with st.spinner("Searching local memory..."):
            # We invoke the LangChain pipeline we built
            response = chain.invoke(user_query)
            st.markdown(response)
    
    # Save response to history
    st.session_state.messages.append({"role": "assistant", "content": response})