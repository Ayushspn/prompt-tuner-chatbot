import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import pipeline

# Setup
embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("knowledge_base")

docs = [
    "Python was created by Guido van Rossum in 1991.",
    "Transformers use attention mechanisms to process sequences.",
    "Bajaj Finance is a major NBFC in India."
]

for i, doc in enumerate(docs):
    emb = embedder.encode([doc]).tolist()
    collection.add(documents=[doc], embeddings=emb, ids=[str(i)])

generator = pipeline("text-generation", model="gpt2")

# Streamlit Chat UI
st.title("💬 Simple RAG Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if query := st.chat_input("Ask me something..."):
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").write(query)

    # Retrieval
    query_emb = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=query_emb, n_results=2)
    context = " ".join(results["documents"][0])

    # Generation
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    result = generator(prompt, max_length=80, do_sample=True)
    answer = result[0]["generated_text"]

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)
