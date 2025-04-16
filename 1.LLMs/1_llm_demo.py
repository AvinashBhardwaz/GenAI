# Import the OpenAI wrapper from Langchain
from langchain_openai import OpenAI

# Import the load_dotenv function to load environment variables from a .env file
from dotenv import load_dotenv

# Load environment variables (like API keys) from a .env file into the environment
load_dotenv()

# Initialize the OpenAI language model using the 'gpt-3.5-turbo-instruct' model
llm = OpenAI(model='gpt-3.5-turbo-instruct')

# Invoke the language model with a query to get the response
result = llm.invoke("What is the capital of India")

# Print the result returned by the model
print(result)
