# Import required modules
from langchain_openai import ChatOpenAI              # OpenAI LLM
from langchain_anthropic import ChatAnthropic        # (Imported but not used here)
from dotenv import load_dotenv                       # To load .env file for API keys
from langchain_core.prompts import PromptTemplate     # Used to define prompt structures
from langchain_core.output_parsers import StrOutputParser  # Basic string output parser
from langchain.schema.runnable import (
    RunnableParallel, RunnableBranch, RunnableLambda  # For advanced chain logic
)
from langchain_core.output_parsers import PydanticOutputParser  # For structured parsing
from pydantic import BaseModel, Field                 # Used to define custom schema
from typing import Literal

# Load API keys and environment variables
load_dotenv()

# Initialize OpenAI model
model = ChatOpenAI()

# Default parser to parse string output from LLM
parser = StrOutputParser()

# Define a Pydantic model for sentiment classification
class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(
        description='Give the sentiment of the feedback'
    )

# Parser that turns LLM output into the structured Feedback object
parser2 = PydanticOutputParser(pydantic_object=Feedback)

# Prompt to classify sentiment with format instructions for structured output
prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into postive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction': parser2.get_format_instructions()}
)

# Sentiment classification chain
classifier_chain = prompt1 | model | parser2

# Prompt to respond to positive feedback
prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

# Prompt to respond to negative feedback
prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

# Conditional branching:
# Depending on the sentiment ('positive' or 'negative'), it runs the appropriate response prompt.
# If neither condition is met (somehow), a fallback message is returned.
branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x: x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "could not find sentiment")  # Fallback
)

# Final chain: classify the sentiment → branch to correct response generation
chain = classifier_chain | branch_chain

# Invoke the chain with a sample positive feedback
print(chain.invoke({'feedback': 'This is a beautiful phone'}))

# Print ASCII diagram of how the chain is constructed
chain.get_graph().print_ascii()
