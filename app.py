import os
from dotenv import load_dotenv
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Load API key
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

# UI Title
st.set_page_config(page_title="Legal AI Chat", layout="wide")
st.title("⚖️ Legal Document Chat AI")

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Upload PDFs
uploaded_files = st.file_uploader(
    "📄 Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    all_docs = []

    # Load PDFs
    for file in uploaded_files:
        with open(file.name, "wb") as f:
            f.write(file.getbuffer())

        loader = PyPDFLoader(file.name)
        docs = loader.load()
        all_docs.extend(docs)

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(all_docs)

    # Embeddings
    embeddings = OpenAIEmbeddings(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )

    # Vector DB
    db = FAISS.from_documents(chunks, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # LLM
    llm = ChatOpenAI(
        model="meta-llama/llama-3-8b-instruct",
        temperature=0.2,
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )

    # Chat UI
    query = st.text_input("💬 Ask your question")

    if query:
        docs = retriever.invoke(query)

        context = "\n".join([doc.page_content for doc in docs])

        prompt = f"""
You are a legal assistant AI.

Answer ONLY from the given context.
If answer is not present, say "Not found in document".

Context:
{context}

Question:
{query}
"""

        response = llm.invoke(prompt)

        # Save chat
        st.session_state.chat_history.append(("You", query))
        st.session_state.chat_history.append(("AI", response.content))

# Display chat history
if st.session_state.chat_history:
    st.subheader("💬 Chat History")

    for role, message in st.session_state.chat_history:
        if role == "You":
            st.markdown(f"🧑 **You:** {message}")
        else:
            st.markdown(f"🤖 **AI:** {message}")