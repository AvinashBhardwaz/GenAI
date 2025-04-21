from langchain_openai import ChatOpenAI  # OpenAI chat model interface
from dotenv import load_dotenv  # To load environment variables from .env
from langchain_core.prompts import PromptTemplate  # For creating reusable prompt templates
from langchain_core.output_parsers import StrOutputParser  # Parses output into a plain string

# Load environment variables (e.g., OpenAI API key)
load_dotenv()

# Step 1: Define a prompt template with a placeholder for 'topic'
prompt = PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic']
)

# Step 2: Initialize the ChatOpenAI model (uses the OpenAI key from .env)
model = ChatOpenAI()

# Step 3: Create a simple output parser to get string output from the model
parser = StrOutputParser()

# Step 4: Compose the chain — prompt feeds into model, which feeds into parser
chain = prompt | model | parser

# Step 5: Invoke the chain with an input topic
result = chain.invoke({'topic': 'cricket'})

# Step 6: Print the final output (i.e., 5 interesting facts about cricket)
print(result)

# Step 7: Visualize the chain structure in ASCII format
chain.get_graph().print_ascii()
