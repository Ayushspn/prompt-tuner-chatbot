from transformers import pipeline

# Load a pre-trained transformer model
# This downloads a small model automatically!
classifier = pipeline("sentiment-analysis")

# Test it!
texts = [
    "Python is amazing for AI!",
    "This product is terrible!",
    "I love building RAG applications!",
    "The code is buggy and slow.",
]

for text in texts:
    result = classifier(text)
    print(f"Text: {text}")
    print(f"Result: {result}")
    print()