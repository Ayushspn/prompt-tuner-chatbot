# Tokenization & Embedding Visualizer
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Define sentences
sentences = [
    "The cat sat on the mat.",
    "A feline rested on a rug.",
    "Cooking pasta is fun.",
]

# 2. Tokenization
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
for s in sentences:
    tokens = tokenizer.tokenize(s)
    print(f"Sentence: {s}")
    print(f"Tokens: {tokens}\n")

# 3. Embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(sentences)

# 4. Cosine similarity matrix
def cosine_sim_matrix(embs):
    normed = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    return np.dot(normed, normed.T)

sim_matrix = cosine_sim_matrix(embeddings)

# 5. Visualization
plt.figure(figsize=(6,4))
sns.heatmap(sim_matrix, annot=True, xticklabels=sentences, yticklabels=sentences, cmap="Blues")
plt.title("Sentence Embedding Similarity")
plt.show()
