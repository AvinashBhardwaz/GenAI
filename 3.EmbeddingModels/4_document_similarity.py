# Import necessary libraries
from langchain_openai import OpenAIEmbeddings  # Importing OpenAIEmbeddings for generating text embeddings
from dotenv import load_dotenv  # Importing load_dotenv to load environment variables from a .env file
from sklearn.metrics.pairwise import cosine_similarity  # Importing cosine_similarity to calculate similarity between embeddings
import numpy as np  # Importing numpy for numerical operations (though not used directly here)

# Load environment variables from the .env file (e.g., API keys)
load_dotenv()

# Initialize the OpenAIEmbeddings class with the model and dimensions
# - model: text-embedding-3-large (a large OpenAI embedding model for generating text embeddings)
# - dimensions: the number of dimensions in the embedding vector (300 in this case)
embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=300)

# List of documents (text) to embed into vectors
documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

# Query for similarity search
query = 'tell me about bumrah'

# Generate embeddings for the documents
doc_embeddings = embedding.embed_documents(documents)

# Generate the embedding for the query
query_embedding = embedding.embed_query(query)

# Calculate cosine similarity between the query embedding and each document's embedding
# cosine_similarity returns a list of similarity scores between the query and each document
scores = cosine_similarity([query_embedding], doc_embeddings)[0]

# Sort the documents by similarity score in descending order, and get the highest score and its index
index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

# Output the query, the most similar document, and the similarity score
print(query)
print(documents[index])  # The document most similar to the query
print("Similarity score is:", score)  # The similarity score between the query and the most similar document
