# Import a loader that reads text files (like .txt)
from langchain_community.document_loaders import TextLoader

# Import the ChatOpenAI model from LangChain's OpenAI integration
from langchain_openai import ChatOpenAI

# Output parser to convert the response from the model into a plain string
from langchain_core.output_parsers import StrOutputParser

# Used to define and format prompts for the LLM
from langchain_core.prompts import PromptTemplate

# Loads environment variables from a .env file (like your OpenAI API key)
from dotenv import load_dotenv

# Load environment variables (needed to authenticate with OpenAI)
load_dotenv()

# What: LLM model from OpenAI (ChatGPT under the hood)
# Why: To generate a response based on our input prompt
# How: Instantiates a default ChatOpenAI model (GPT-3.5 or 4 depending on your config)
model = ChatOpenAI()

# What: A prompt template
# Why: To structure the input to the LLM clearly and dynamically
# How: Uses {poem} as a placeholder that gets replaced at runtime
prompt = PromptTemplate(
    template='Write a summary for the following poem - \n {poem}',
    input_variables=['poem']
)

# What: Output parser
# Why: Extracts the raw string output from the model's structured response
# How: Converts LLM result to a plain string
parser = StrOutputParser()

# What: Loads a .txt file
# Why: To read a text file and convert it into LangChain documents
# How: Uses UTF-8 encoding to handle special characters properly
loader = TextLoader('cricket.txt', encoding='utf-8')

# What: Loads the text content
# Why: Required to convert file content into LangChain-compatible format (Document)
# How: Returns a list of Document objects with content and metadata
docs = loader.load()

# Print type of the loaded documents (for understanding/debugging)
print(type(docs))  # Should show <class 'list'>

# Print how many document chunks were loaded (usually 1 for a short .txt)
print(len(docs))

# Print the actual poem text content from the first document
print(docs[0].page_content)

# Print any metadata like file path or line number
print(docs[0].metadata)

# What: Creates a processing pipeline (RunnableSequence)
# Why: To modularly pass the prompt → model → parser
# How: Uses LangChain's pipe operator | for chaining steps
chain = prompt | model | parser

# What: Invokes the chain
# Why: To pass the poem text and get a summary
# How: Fills prompt with the poem, sends to LLM, parses the response
print(chain.invoke({'poem': docs[0].page_content}))
