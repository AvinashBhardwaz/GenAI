# Import necessary libraries
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env file (e.g., OpenAI API key)
load_dotenv()

# Define the first prompt template: this will generate a detailed report on a given topic
prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',  # Template string
    input_variables=['topic']  # The input variable for the template, i.e., topic (e.g., 'Unemployment in India')
)

# Define the second prompt template: this will generate a 5-point summary of the text passed as input
prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',  # Template string
    input_variables=['text']  # The input variable for the template, i.e., the text to summarize
)

# Initialize the ChatOpenAI model from OpenAI's API
model = ChatOpenAI()

# Initialize the output parser to extract the plain text result from the model's output
parser = StrOutputParser()

# Create a chain of prompts and models:
# 1. First, the detailed report is generated using prompt1 and model.
# 2. Then, the output is parsed and passed to prompt2 for summarization.
# 3. Finally, the second model generates a 5-point summary and the output is parsed.
chain = prompt1 | model | parser | prompt2 | model | parser

# Invoke the chain with the input {'topic': 'Unemployment in India'}
# This will first generate the report and then summarize it in 5 points.
result = chain.invoke({'topic': 'Unemployment in India'})

# Print the final summarized result (5-point summary)
print(result)

# Print the ASCII graph of the chain of components (shows how data flows through the chain)
chain.get_graph().print_ascii()
