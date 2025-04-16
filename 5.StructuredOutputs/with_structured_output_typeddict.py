from langchain_openai import ChatOpenAI  # Importing the OpenAI-based model from LangChain
from dotenv import load_dotenv  # To load environment variables from .env file
from typing import TypedDict, Annotated, Optional, Literal  # Importing necessary types for type hinting

# Load environment variables from a .env file
load_dotenv()

# Initialize the OpenAI model from LangChain
model = ChatOpenAI()

# Define the schema for the review using TypedDict
class Review(TypedDict):
    # Key themes discussed in the review (list of strings)
    key_themes: Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
    
    # A brief summary of the review (string)
    summary: Annotated[str, "A brief summary of the review"]
    
    # Sentiment of the review (either 'pos' for positive, 'neg' for negative)
    sentiment: Annotated[Literal["pos", "neg"], "Return sentiment of the review either negative, positive or neutral"]
    
    # List of pros (optional list of strings)
    pros: Annotated[Optional[list[str]], "Write down all the pros inside a list"]
    
    # List of cons (optional list of strings)
    cons: Annotated[Optional[list[str]], "Write down all the cons inside a list"]
    
    # Name of the reviewer (optional string)
    name: Annotated[Optional[str], "Write the name of the reviewer"]

# Create a structured model with the defined Review schema
structured_model = model.with_structured_output(Review)

# Input review text for analysis
result = structured_model.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by Avinash Bhardwaz
""")

# Print the name of the reviewer from the structured output
print(result['name'])  # Output: Avinash Bhardwaz
