# Import the ChatOpenAI class from the langchain_openai module
from langchain_openai import ChatOpenAI

# Import load_dotenv to load environment variables from a .env file
from dotenv import load_dotenv

# Load environment variables (e.g., OpenAI API key) from the .env file into the environment
load_dotenv()

# Initialize the ChatOpenAI model with specific parameters:
# - model='gpt-4': use GPT-4 model
# - temperature=1.5: higher creativity in responses
# - max_completion_tokens=10: limit the number of tokens in the response
model = ChatOpenAI(model='gpt-4', temperature=1.5, max_completion_tokens=10)

# Send a prompt to the model to generate a response
result = model.invoke("Write a 5 line poem on cricket")

# Print only the content (text) part of the model's response
print(result.content)
