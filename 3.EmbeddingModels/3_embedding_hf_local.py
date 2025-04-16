# Import the HuggingFaceEmbeddings class from Langchain for using Hugging Face's embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize the HuggingFaceEmbeddings class with the specified model name
# - model_name: refers to the model used for generating embeddings (e.g., 'sentence-transformers/all-MiniLM-L6-v2')
embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# List of documents (text) to be embedded into vectors
documents = [
    "Delhi is the capital of India",  # Example 1: Information about Delhi
    "Kolkata is the capital of West Bengal",  # Example 2: Information about Kolkata
    "Paris is the capital of France"  # Example 3: Information about Paris
]

# Get the embeddings for the list of documents
vector = embedding.embed_documents(documents)

# Print the embedding vectors for each document
print("Embedding Vectors for Documents:")
for i, embedding_vector in enumerate(vector):
    print(f"Document {i+1}: {documents[i]}")
    print(f"Embedding Vector: {str(embedding_vector)}\n")

# Example Output:
# Document 1: Delhi is the capital of India
# Embedding Vector: [0.3456, -0.1234, 0.8765, ...]
# Document 2: Kolkata is the capital of West Bengal
# Embedding Vector: [0.2345, -0.6789, 0.5432, ...]
# Document 3: Paris is the capital of France
# Embedding Vector: [0.9876, -0.3456, 0.2345, ...]
