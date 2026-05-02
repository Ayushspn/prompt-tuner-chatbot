import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import pipeline

# Step 1: Load Knowledge Base dynamically
with open("knowledge.txt", "r", encoding="utf-8") as f:
    docs = [line.strip() for line in f if line.strip()]

# Step 2: Embedding Model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Step 3: ChromaDB Setup
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("knowledge_base")

# Add documents with embeddings
for i, doc in enumerate(docs):
    emb = embedder.encode([doc]).tolist()
    collection.add(documents=[doc], embeddings=emb, ids=[str(i)])

# Step 4: Streamlit UI
st.title("📚 Dynamic RAG Demo")
query = st.text_input("Enter your question:")

if query:
    # Retrieval
    query_emb = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=query_emb, n_results=2)
    context = " ".join(results["documents"][0])

    # Generator
    generator = pipeline("text-generation", model="gpt2")
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    result = generator(prompt, max_length=80, do_sample=True)

    st.subheader("Retrieved Context")
    st.write(context)

    st.subheader("Generated Answer")
    st.write(result[0]['generated_text'])
