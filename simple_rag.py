from sentence_transformers import SentenceTransformer
import chromadb
from transformers import pipeline

# Step 1: Knowledge Base
docs = [
    "Python was created by Guido van Rossum in 1991.",
    "Transformers use attention mechanisms to process sequences.",
    "Bajaj Finance is a major NBFC in India."
]

# Step 2: Embedding Model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Step 3: ChromaDB Setup
client = chromadb.Client()
collection = client.create_collection("knowledge_base")

# Add documents with embeddings
for i, doc in enumerate(docs):
    emb = embedder.encode([doc]).tolist()
    collection.add(documents=[doc], embeddings=emb, ids=[str(i)])

# Step 4: Query (take input from user)
query = input("Enter your question: ")
query_emb = embedder.encode([query]).tolist()

# Step 5: Retrieval
results = collection.query(query_embeddings=query_emb, n_results=1)
retrieved = results["documents"][0][0]

# Step 6: Generator (LLM)
generator = pipeline("text-generation", model="gpt2")

prompt = f"Context: {retrieved}\nQuestion: {query}\nAnswer:"
result = generator(prompt, max_length=50)
print(result[0]['generated_text'])
