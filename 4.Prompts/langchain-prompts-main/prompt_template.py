# Import PromptTemplate to define prompt structure
from langchain_core.prompts import PromptTemplate

# Import ChatOpenAI to interact with OpenAI's chat models
from langchain_openai import ChatOpenAI

# Import dotenv to load environment variables from a .env file
from dotenv import load_dotenv

# Load environment variables (like OpenAI API key) from .env
load_dotenv()

# Initialize the OpenAI chat model (defaults to gpt-3.5-turbo)
model = ChatOpenAI()

# --------------------------
# Define the Prompt Template
# --------------------------
template2 = PromptTemplate(
    template='Greet this person in 5 languages. The name of the person is {name}',  # Prompt with a placeholder
    input_variables=['name']  # Declare which variable(s) will be substituted
)

# --------------------------
# Fill the prompt with data
# --------------------------
prompt = template2.invoke({'name': 'nitish'})  # Replace `{name}` with 'nitish'

# --------------------------
# Send the prompt to the model
# --------------------------
result = model.invoke(prompt)  # Pass the filled-in prompt to the OpenAI model

# --------------------------
# Print the model's response
# --------------------------
print(result.content)  # Output the generated greetings
