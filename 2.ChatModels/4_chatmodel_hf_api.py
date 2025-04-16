# Import required classes for using Hugging Face models with Langchain
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Import load_dotenv to load environment variables from a .env file
from dotenv import load_dotenv

# Load environment variables (like Hugging Face API token) from the .env file
load_dotenv()

# Initialize the Hugging Face model endpoint
# - repo_id: points to the TinyLlama chat model hosted on Hugging Face
# - task: specifies the type of task, in this case, text generation
llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)

# Wrap the model with ChatHuggingFace for a chat-style interface
model = ChatHuggingFace(llm=llm)

# Send a prompt to the model and receive the generated response
result = model.invoke("What is the capital of India")

# Print only the content of the response
print(result.content)
