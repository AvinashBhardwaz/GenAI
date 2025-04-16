# Import the ChatAnthropic class from the langchain_anthropic module
from langchain_anthropic import ChatAnthropic

# Import load_dotenv to load environment variables from a .env file
from dotenv import load_dotenv

# Load environment variables (like the Anthropic API key) from the .env file
load_dotenv()

# Initialize the Claude model from Anthropic using the 'claude-3-5-sonnet-20241022' version
model = ChatAnthropic(model='claude-3-5-sonnet-20241022')

# Send a prompt to the model asking for the capital of India
result = model.invoke('What is the capital of India')

# Print the content part of the response returned by the Claude model
print(result.content)
