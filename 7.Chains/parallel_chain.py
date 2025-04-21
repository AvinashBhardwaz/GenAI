# Import required libraries and modules
from langchain_openai import ChatOpenAI  # For using OpenAI's LLMs
from langchain_anthropic import ChatAnthropic  # For using Anthropic's Claude models
from dotenv import load_dotenv  # To load environment variables from .env
from langchain_core.prompts import PromptTemplate  # For creating structured prompts
from langchain_core.output_parsers import StrOutputParser  # To parse raw string output
from langchain.schema.runnable import RunnableParallel  # To run multiple chains in parallel

# Load environment variables like API keys
load_dotenv()

# Initialize OpenAI model (e.g., GPT-4)
model1 = ChatOpenAI()

# Initialize Anthropic model (e.g., Claude 3 Sonnet)
model2 = ChatAnthropic(model_name='claude-3-7-sonnet-20250219')

# Prompt to generate simple notes from input text
prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)

# Prompt to generate 5 short Q&A from the same input text
prompt2 = PromptTemplate(
    template='Generate 5 short question answers from the following text \n {text}',
    input_variables=['text']
)

# Prompt to merge notes and quiz into a single final document
prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)

# Output parser to convert LLM output into clean strings
parser = StrOutputParser()

# Create a parallel chain that runs both note and quiz generation at the same time
parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,  # Notes generation using OpenAI
    'quiz': prompt2 | model2 | parser    # Quiz generation using Claude
})

# Chain to merge the parallel outputs into one final document
merge_chain = prompt3 | model1 | parser

# Combine the full pipeline: parallel generation -> merging
chain = parallel_chain | merge_chain

# Input text about Support Vector Machines (SVMs)
text = """
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.
"""

# Invoke the full chain with the text input
result = chain.invoke({'text': text})

# Print the final merged result (notes + quiz)
print(result)

# Show the flow graph of the chain (useful for debugging and visualization)
chain.get_graph().print_ascii()
