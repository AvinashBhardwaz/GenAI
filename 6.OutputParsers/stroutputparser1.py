from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables (e.g., OpenAI API key)
load_dotenv()

# Initialize OpenAI chat model (defaults to gpt-3.5 or as configured)
model = ChatOpenAI()

# First prompt: ask for a detailed report on a given topic
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

# Second prompt: summarize the above report in 5 lines
template2 = PromptTemplate(
    template='Write a 5 line summary on the following text.\n{text}',  # Fixed newline
    input_variables=['text']
)

# String output parser to cleanly extract string from LLM output
parser = StrOutputParser()

# Chain: prompt1 → model → parser → prompt2 → model → parser
# This creates a linear flow where:
# 1. A report is generated
# 2. It's parsed into plain text
# 3. That text is passed to the summary prompt
# 4. Summary is generated and parsed again
chain = template1 | model | parser | template2 | model | parser

# Run the chain with a topic
result = chain.invoke({'topic': 'black hole'})

# Final summarized output
print(result)
