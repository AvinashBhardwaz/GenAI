from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

# Load environment variables from .env file (e.g., HuggingFace token)
load_dotenv()

# Define the LLM from HuggingFace (Gemma 2B Instruct model)
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",  # HuggingFace model repo
    task="text-generation"          # Task type
)

# Wrap the model in ChatHuggingFace for chat-style usage
model = ChatHuggingFace(llm=llm)

# 1st PromptTemplate: Generates a detailed report on a given topic
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

# 2nd PromptTemplate: Summarizes any given text in 5 lines
template2 = PromptTemplate(
    template='Write a 5 line summary on the following text.\n{text}',  # Fixed: \n instead of /n
    input_variables=['text']
)

# Fill the 1st prompt with the topic 'black hole'
prompt1 = template1.invoke({'topic': 'black hole'})

# Pass the generated report to the model
result = model.invoke(prompt1)

# Fill the 2nd prompt using the output (content) of the 1st response
prompt2 = template2.invoke({'text': result.content})

# Generate the 5-line summary using the model
result1 = model.invoke(prompt2)

# Print the final summarized output
print(result1.content)
