# Import WebBaseLoader to load and scrape text content from a webpage
from langchain_community.document_loaders import WebBaseLoader

# Import ChatOpenAI model to generate responses
from langchain_openai import ChatOpenAI

# Import output parser to extract raw string output from the LLM
from langchain_core.output_parsers import StrOutputParser

# Import PromptTemplate to define dynamic input-based prompts
from langchain_core.prompts import PromptTemplate

# Load environment variables (like your OpenAI API key from .env file)
from dotenv import load_dotenv
load_dotenv()

# What: Create an LLM object using OpenAI
# Why: We'll use this to generate answers to questions based on webpage content
# How: Using the `ChatOpenAI()` constructor which pulls credentials from env vars
model = ChatOpenAI()

# What: Define a prompt template for asking questions based on given text
# Why: To instruct the LLM on how to answer questions using the provided content
# How: Use placeholders {question} and {text} to be filled dynamically at runtime
prompt = PromptTemplate(
    template='Answer the following question \n {question} from the following text - \n {text}',
    input_variables=['question','text']
)

# What: Create a parser to convert LLM output into plain string
# Why: By default, LLM responses are wrapped; we want clean string output
# How: StrOutputParser removes the structured metadata and returns raw string
parser = StrOutputParser()

# What: URL of the product page we want to load and analyze
# Why: To use real-world content for answering questions
# How: Define the product page link from Flipkart
url = 'https://www.flipkart.com/apple-macbook-air-m2-16-gb-256-gb-ssd-macos-sequoia-mc7x4hn-a/p/itmdc5308fa78421'

# What: Initialize WebBaseLoader with the product URL
# Why: WebBaseLoader fetches and parses web content into document format
# How: Pass the URL to the loader
loader = WebBaseLoader(url)

# What: Load and extract content from the web page
# Why: To retrieve the text that the LLM will use to answer questions
# How: .load() fetches and returns a list of Document objects
docs = loader.load()

# What: Create a chain by connecting prompt -> model -> parser
# Why: To automate passing input through each component in sequence
# How: Use `|` operator (pipe) to build the sequence chain
chain = prompt | model | parser

# What: Ask a question using the extracted text as context
# Why: To demonstrate how LLM can answer based on scraped web content
# How: Provide a dictionary with values for {question} and {text} in the prompt
print(chain.invoke({
    'question': 'What is the product that we are talking about?', 
    'text': docs[0].page_content
}))
