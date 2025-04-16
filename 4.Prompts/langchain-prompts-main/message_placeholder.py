# Import required classes
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Create a ChatPromptTemplate to define the structure of the prompt
chat_template = ChatPromptTemplate([
    # System message sets the assistant's behavior
    ('system', 'You are a helpful customer support agent'),
    
    # MessagesPlaceholder dynamically injects the chat history into the prompt
    # Instead of hardcoding past interactions, this allows injecting a list of messages (System/Human/AI)
    MessagesPlaceholder(variable_name='chat_history'),
    
    # Human message takes the latest query from the user
    ('human', '{query}')
])

# Initialize an empty chat history list
chat_history = []

# Load chat history from a file and extend the list
# This assumes chat history is stored as text lines, but ideally, it should be stored as serialized messages
with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())  # NOTE: lines are strings, not message objects

# Print the loaded chat history (for debugging)
print(chat_history)

# Create the prompt using the template by filling in placeholders:
# - 'chat_history': dynamically inserted conversation messages (via MessagesPlaceholder)
# - 'query': the current user question
prompt = chat_template.invoke({
    'chat_history': chat_history,
    'query': 'Where is my refund'
})

# Print the final constructed prompt
print(prompt)
