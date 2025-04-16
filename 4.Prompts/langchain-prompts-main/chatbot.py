# Importing the necessary classes and functions
from langchain_openai import ChatOpenAI  # ChatOpenAI allows interaction with OpenAI's chat models
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage  # Message types to build conversation history
from dotenv import load_dotenv  # Used to load environment variables from a .env file (e.g., API keys)

# Load the environment variables
load_dotenv()

# Initialize the chat model (default: gpt-3.5-turbo)
model = ChatOpenAI()

# Initialize chat history with a system message to set the behavior of the AI
chat_history = [
    SystemMessage(content='You are a helpful AI assistant')
]

# Start a loop to have a back-and-forth chat with the AI
while True:
    # Take user input from the terminal
    user_input = input('You: ')
    
    # Append the user's message to the chat history
    chat_history.append(HumanMessage(content=user_input))
    
    # Exit the loop if the user types 'exit'
    if user_input == 'exit':
        break
    
    # Send the full chat history to the model to maintain context
    result = model.invoke(chat_history)
    
    # Append the AI's response to the chat history
    chat_history.append(AIMessage(content=result.content))
    
    # Print the AI's reply
    print("AI: ", result.content)

# Print the complete conversation history once the loop ends
print(chat_history)
