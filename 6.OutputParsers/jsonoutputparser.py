from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Load environment variables (like API keys) from .env file
load_dotenv()

# Define the LLM using HuggingFaceEndpoint (Gemma-2B-Instruct model in this case)
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",  # Model repo on HuggingFace
    task="text-generation"           # Task type
)

# Wrap the endpoint in ChatHuggingFace to enable chat-like interface
model = ChatHuggingFace(llm=llm)

# Define the output parser — this will parse the string output as JSON
parser = JsonOutputParser()

# Create a prompt template that takes a 'topic' and formats instructions from the parser
template = PromptTemplate(
    template='Give me 5 facts about {topic} \n {format_instruction}',  # Prompt with format instruction
    input_variables=['topic'],                                         # Variables to be filled at runtime
    partial_variables={'format_instruction': parser.get_format_instructions()}  # JSON format hint
)

# Build the chain: Prompt → Model → JSON Output Parser
chain = template | model | parser

# Invoke the chain with a specific topic
result = chain.invoke({'topic':'black hole'})

# Print the parsed JSON result
print(result)
