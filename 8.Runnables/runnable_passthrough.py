from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough

# Load environment variables from .env file (useful for loading API keys)
load_dotenv()

# --- PROMPT TEMPLATES ---

# Prompt to generate a joke on a given topic
prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

# Prompt to explain the joke that was generated
prompt2 = PromptTemplate(
    template='Explain the following joke - {text}',
    input_variables=['text']
)

# --- MODEL AND PARSER ---

# Language model to generate or process text (here using OpenAI's chat model)
model = ChatOpenAI()

# Parser to extract plain text from the model's output
parser = StrOutputParser()

# --- JOKE GENERATION CHAIN ---

# RunnableSequence: Chains the joke generation steps
# This chain runs: prompt1 -> model -> parser
joke_gen_chain = RunnableSequence(prompt1, model, parser)

# --- PARALLEL CHAIN FOR EXPLANATION ---

# RunnablePassthrough: Simply passes input to next stage without modifying it.
# Why use it here? Because we need to pass the joke *as-is* to one part ("joke")
# and also send it for further explanation in another chain ("explanation")

# RunnableParallel runs both branches in parallel and returns a dictionary of results
# joke -> unchanged joke text (for display), explanation -> explanation from model
parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),  # Just passes the joke output directly
    'explanation': RunnableSequence(prompt2, model, parser)  # Explains the joke
})

# --- FINAL CHAIN ---

# This is the full chain:
# Step 1: Generate the joke
# Step 2: In parallel: return the joke and generate an explanation
final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

# Run the chain with topic = "cricket"
print(final_chain.invoke({'topic': 'cricket'}))

