from transformers import BertTokenizer, BertModel
import torch
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load model + tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)

# 2. Input sentence
sentence = "The cat sat on the mat."
inputs = tokenizer(sentence, return_tensors="pt")

# 3. Forward pass with attention
outputs = model(**inputs)
attentions = outputs.attentions  # list of attention matrices per layer

# 4. Pick one layer & one head
layer = 0
head = 0
attn_matrix = attentions[layer][0, head].detach().numpy()

# 5. Visualization
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
plt.figure(figsize=(8,6))
sns.heatmap(attn_matrix, xticklabels=tokens, yticklabels=tokens, cmap="Blues")
plt.title(f"Attention Map - Layer {layer+1}, Head {head+1}")
plt.show()
