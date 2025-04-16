# Import necessary classes for using Hugging Face models with Langchain
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os

# Set the Hugging Face cache directory to store models and data locally
os.environ['HF_HOME'] = 'D:/huggingface_cache'

# Initialize the Hugging Face pipeline using the specified model
# - model_id: refers to the model 'TinyLlama-1.1B-Chat-v1.0' hosted on Hugging Face
# - task: specifies the task, in this case, 'text-generation'
# - pipeline_kwargs: additional settings like temperature and max token generation
llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',  # The model identifier
    task='text-generation',  # The type of task the model will perform
    pipeline_kwargs=dict(   # Additional parameters for text generation
        temperature=0.5,  # Controls randomness (0.0 = deterministic, 1.0 = more random)
        max_new_tokens=100  # Maximum number of tokens to generate in the response
    )
)

# Wrap the model into a ChatHuggingFace instance for easier interaction
model = ChatHuggingFace(llm=llm)

# Send a prompt to the model and get the generated response
result = model.invoke("What is the capital of India")

# Print the content (text) part of the result returned by the model
print(result.content)
