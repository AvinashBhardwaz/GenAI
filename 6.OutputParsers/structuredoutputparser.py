from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# Load environment variables (like HuggingFace API key if in .env file)
load_dotenv()

# Initialize HuggingFace endpoint with a specific model
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",  # Using Google’s Gemma 2B Instruct model
    task="text-generation"
)

# Wrap the model for chat-based use
model = ChatHuggingFace(llm=llm)

# Define the structure of the expected response (schema)
schema = [
    ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about the topic'),
]

# Create a structured output parser using the schema
parser = StructuredOutputParser.from_response_schemas(schema)

# Define the prompt with placeholders for topic and formatting instructions
template = PromptTemplate(
    template='Give 3 fact about {topic} \n {format_instruction}',  # Format instruction guides model response
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}  # Inject format rules
)

# Build the chain: prompt → model → parser
chain = template | model | parser

# Run the chain with a specific topic
result = chain.invoke({'topic': 'black hole'})

# Print the structured result (dict with keys: fact_1, fact_2, fact_3)
print(result)

"""
Sample Output (structured dictionary):
{
    'fact_1': 'A black hole is a region of space where gravity is so strong that nothing can escape.',
    'fact_2': 'Black holes can be formed when massive stars collapse at the end of their life cycles.',
    'fact_3': 'The boundary around a black hole is called the event horizon.'
}
"""
