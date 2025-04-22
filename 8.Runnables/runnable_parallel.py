# Importing required modules from langchain and other dependencies
from langchain_openai import ChatOpenAI  # For interacting with the OpenAI API
from langchain_core.prompts import PromptTemplate  # To define templates for prompts
from langchain_core.output_parsers import StrOutputParser  # To parse the output of the model as strings
from dotenv import load_dotenv  # To load environment variables from .env file
from langchain.schema.runnable import RunnableSequence, RunnableParallel  # For creating sequential and parallel chains of tasks

# Loading environment variables, such as API keys, from a .env file
load_dotenv()

# Define the first prompt template for generating a tweet
prompt1 = PromptTemplate(
    template='Generate a tweet about {topic}',  # Template for tweet generation
    input_variables=['topic']  # The input variable in the template
)

# Define the second prompt template for generating a LinkedIn post
prompt2 = PromptTemplate(
    template='Generate a Linkedin post about {topic}',  # Template for LinkedIn post generation
    input_variables=['topic']  # The input variable in the template
)

# Initialize the OpenAI model for generating responses (using GPT model under the hood)
model = ChatOpenAI()

# Initialize a parser to convert the model's raw output into a string format
parser = StrOutputParser()

# Define a parallel chain of tasks where both the tweet and the LinkedIn post are generated at the same time
parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(prompt1, model, parser),  # A sequence for generating a tweet
    'linkedin': RunnableSequence(prompt2, model, parser)  # A sequence for generating a LinkedIn post
})

# Invoke the parallel chain by passing a topic as input (in this case, 'AI')
result = parallel_chain.invoke({'topic':'AI'})

# Print the generated tweet and LinkedIn post from the result
print(result['tweet'])  # Output the generated tweet
print(result['linkedin'])  # Output the generated LinkedIn post
