from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import (
    RunnableSequence,        # To run multiple components in sequence
    RunnableParallel,        # (Not used here but part of import)
    RunnablePassthrough,     # To pass input through unchanged
    RunnableBranch,          # To choose between logic paths based on a condition
    RunnableLambda           # (Not used here but part of import)
)

load_dotenv()  # Load environment variables from a .env file (e.g., for OpenAI API keys)

# ------------------ PROMPT DEFINITIONS ------------------

# PromptTemplate defines a text template with placeholders
prompt1 = PromptTemplate(
    template='Write a detailed report on {topic}',   # Template to generate detailed report
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the following text \n {text}',   # Template to summarize a long report
    input_variables=['text']
)

# ------------------ MODEL AND PARSER ------------------

model = ChatOpenAI()               # OpenAI chat model for text generation
parser = StrOutputParser()         # Converts raw model output to plain string

# ------------------ CHAIN: Report Generation ------------------

# This chain generates a detailed report from a topic
# Uses pipe operator (|) to pass output from one step to the next
report_gen_chain = prompt1 | model | parser

# ------------------ BRANCH LOGIC ------------------

# RunnableBranch:
# What: Allows choosing different chains based on a condition
# Why: To perform dynamic behavior depending on data (e.g., summarize if too long)
# How: Accepts (condition, runnable) pairs. If condition is True, it runs that branch.

# RunnablePassthrough:
# What: A utility Runnable that just returns the input as-is
# Why: Acts as the "else" or fallback when no conditions are met
# How: Useful for default behavior in branching logic

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 300, prompt2 | model | parser),  # If generated report is too long, summarize it
    RunnablePassthrough()                                        # Else, return the report as-is
)

# ------------------ FINAL CHAIN ------------------

# RunnableSequence:
# What: Executes a list of runnables in order
# Why: Helps modularize and control flow of logic clearly
# How: First generates report, then conditionally summarizes

final_chain = RunnableSequence(report_gen_chain, branch_chain)

# ------------------ EXECUTION ------------------

# Run the final chain on the given topic
print(final_chain.invoke({'topic':'Russia vs Ukraine'}))
