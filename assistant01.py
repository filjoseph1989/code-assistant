from sentence_transformers import SentenceTransformer
import torch
import os

# This environment variable silences the tokenizer parallelism warning.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Download from the 🤗 Hub
model = SentenceTransformer("google/embeddinggemma-300m")

# Run inference with queries and documents
query = "Which planet is known as the Red Planet?"
print(f"Query: \"{query}\"\n")

documents = [
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet."
]

# The .encode() method is the modern way to handle both single strings and lists.
query_embedding = model.encode(query)
document_embeddings = model.encode(documents)

print("Shape of query embedding:", query_embedding.shape)
print("Shape of document embeddings:", document_embeddings.shape)

# Compute similarities to determine a ranking
similarities = model.similarity(query_embedding, document_embeddings)
print("\nSimilarity scores (Query vs. each Document):", similarities)

# Find and print the best match
best_match_index = torch.argmax(similarities)
print(f"\n---> Best match found: '{documents[best_match_index]}'")
print(f"---> Similarity score: {similarities[0][best_match_index]:.4f}")
