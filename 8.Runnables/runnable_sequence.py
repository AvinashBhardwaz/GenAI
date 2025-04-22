from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate  # Updated import
from langchain.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema import RunnableSequence

# Load environment variables
load_dotenv()

# Define the prompt for generating a joke
prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

# Initialize OpenAI Chat model
model = ChatOpenAI()

# Define the output parser for string-based output
parser = StrOutputParser()

# Define the second prompt for explaining the joke
prompt2 = PromptTemplate(
    template='Explain the following joke - {text}',
    input_variables=['text']
)

# Construct the chain using RunnableSequence
chain = RunnableSequence(
    [prompt1, model, parser, prompt2, model, parser]
)

# Invoke the chain with input
result = chain.invoke({'topic': 'AI'})

# Print the final result
print(result)
