# Import the OpenAIEmbeddings class from Langchain for using OpenAI's embeddings
from langchain_openai import OpenAIEmbeddings

# Import load_dotenv to load environment variables (like API keys) from a .env file
from dotenv import load_dotenv

# Load environment variables from the .env file into the environment
load_dotenv()

# Initialize the OpenAIEmbeddings class with the specified model and dimensions
# - model: refers to the model used for generating embeddings (e.g., 'text-embedding-3-large')
# - dimensions: specifies the size of the embedding vector (32 in this case)
embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

# Example: Embed a query (text) into an embedding vector
# Here we are using the statement: "Delhi is the capital of India"
query = "Delhi is the capital of India"

# Get the embedding for the query
result = embedding.embed_query(query)

# Print the original query and its corresponding embedding
print("Original Query: ", query)
print("Embedding Vector: ", str(result))

# Example of what the output may look like (an embedding vector is typically a list of floating-point numbers)
# Example Output (simplified):
# Original Query:  Delhi is the capital of India
# Embedding Vector:  [0.1234, -0.5678, 0.3456, ..., 0.9876]  # An example of a vector representation
