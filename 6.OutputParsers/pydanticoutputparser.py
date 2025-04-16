from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Load environment variables (e.g., HF API key, if set via .env)
load_dotenv()

# Initialize the HuggingFace endpoint with a specific model
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",  # Using Gemma 2B Instruct model
    task="text-generation"
)

# Wrap the endpoint with ChatHuggingFace for chat-based interactions
model = ChatHuggingFace(llm=llm)

# Define a Pydantic model to validate and structure the expected output
class Person(BaseModel):
    name: str = Field(description='Name of the person')
    age: int = Field(gt=18, description='Age of the person')  # Age must be > 18
    city: str = Field(description='Name of the city the person belongs to')

# Create a parser that uses the Pydantic model for structured output
parser = PydanticOutputParser(pydantic_object=Person)

# Create a prompt template with a dynamic 'place' input and static format instructions
template = PromptTemplate(
    template='Generate the name, age and city of a fictional {place} person \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction': parser.get_format_instructions()}  # Auto-insert format guideline
)

# Chain: prompt → model → output parser
chain = template | model | parser

# Invoke the chain with a specific place (e.g., Sri Lankan)
final_result = chain.invoke({'place': 'sri lankan'})

# Print the structured result
print(final_result)
