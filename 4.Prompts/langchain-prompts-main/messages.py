# Import message types to simulate a conversation with roles (system, human, AI)
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Import the OpenAI chat model wrapper from LangChain
from langchain_openai import ChatOpenAI

# Load environment variables (to access your OpenAI API key securely from a .env file)
from dotenv import load_dotenv
load_dotenv()

# Initialize the ChatOpenAI model (defaults to 'gpt-3.5-turbo' unless specified)
model = ChatOpenAI()

# Create a list of messages to simulate the start of a conversation
messages = [
    SystemMessage(content='You are a helpful assistant'),  # Set the assistant's behavior
    HumanMessage(content='Tell me about LangChain')         # User's query
]

# Invoke the model with the current conversation messages
result = model.invoke(messages)

# Append the AI's response to the messages to maintain conversation history
messages.append(AIMessage(content=result.content))

# Print the full list of messages (now includes System, Human, and AI response)
print(messages)
