# Importing ChatPromptTemplate from LangChain Core
from langchain_core.prompts import ChatPromptTemplate

# Creating a chat prompt template with roles and placeholders
# - 'system': Provides system-level instructions (like setting the AI's role)
# - 'human': Represents the user's input
chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} expert'),  # This sets the system message using a domain placeholder
    ('human', 'Explain in simple terms, what is {topic}')  # This sets the user's message with a topic placeholder
])

# Filling in the placeholders 'domain' and 'topic' with actual values
# - domain: 'cricket' (so the AI behaves like a cricket expert)
# - topic: 'Dusra' (a term in cricket referring to a delivery type by an off-spin bowler)
prompt = chat_template.invoke({'domain': 'cricket', 'topic': 'Dusra'})

# Printing the final prompt object (will show how the prompt will be passed to the model)
print(prompt)
