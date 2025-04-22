from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableLambda, RunnablePassthrough, RunnableParallel

load_dotenv()  # Loads environment variables from the `.env` file (e.g., OpenAI API key)

# Simple utility function to count words in a string
def word_count(text):
    return len(text.split())

# Creating a prompt that asks the model to write a joke on a given topic
prompt = PromptTemplate(
    template='Write a joke about {topic}',  # Template with a placeholder for the topic
    input_variables=['topic']  # Specifies that 'topic' is a variable to be filled
)

# Instantiating the ChatOpenAI model (OpenAI LLM)
model = ChatOpenAI()

# Output parser to convert model's output into a plain string
parser = StrOutputParser()

# ---------------------- CHAIN 1 ----------------------
# joke_gen_chain combines:
# PromptTemplate ➝ ChatOpenAI ➝ StrOutputParser
# This generates a joke and parses it as plain text

joke_gen_chain = RunnableSequence(prompt, model, parser)

# ---------------------- CHAIN 2 ----------------------
# This chain performs two tasks in parallel:
# 1. Returns the original joke using RunnablePassthrough
# 2. Counts the words in the joke using RunnableLambda

# RunnablePassthrough:
# - WHAT: A simple passthrough node that returns the input as-is.
# - WHY: We use it here to return the original joke without modifying it.
# - HOW: It just passes the joke through into the final output unchanged.

# RunnableLambda:
# - WHAT: A wrapper around a Python function to make it "runnable" in LangChain.
# - WHY: We use it to integrate a custom function (`word_count`) as a part of the chain.
# - HOW: It takes the joke text and applies the `word_count` function to return word count.

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),  # Returns the joke without changing it
    'word_count': RunnableLambda(word_count)  # Calculates word count of the joke
})

# ---------------------- FINAL CHAIN ----------------------
# Combines both joke generation and word count:
# Step 1: Generate joke → Step 2: In parallel, return joke & count its words

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

# Invoking the chain with a topic, returns both joke and its word count
result = final_chain.invoke({'topic': 'AI'})

# Printing the final output in a formatted string
final_result = """{} \n word count - {}""".format(result['joke'], result['word_count'])

print(final_result)
