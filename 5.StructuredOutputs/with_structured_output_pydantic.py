# Import the ChatOpenAI model from LangChain
from langchain_openai import ChatOpenAI

# Load environment variables from a .env file (e.g., for OpenAI API key)
from dotenv import load_dotenv

# Typing imports (some not used but included here as original)
from typing import TypedDict, Annotated, Optional, Literal

# Import Pydantic's BaseModel and Field for schema definition
from pydantic import BaseModel, Field

# Load the .env file (usually contains API keys and config variables)
load_dotenv()

# Initialize the ChatOpenAI model (default settings used)
model = ChatOpenAI()

# Define the structure of the output using a Pydantic model
class Review(BaseModel):
    key_themes: list[str] = Field(description="Write down all the key themes discussed in the review in a list")
    summary: str = Field(description="A brief summary of the review")
    sentiment: Literal["pos", "neg"] = Field(description="Return sentiment of the review either negative, positive or neutral")
    pros: Optional[list[str]] = Field(default=None, description="Write down all the pros inside a list")
    cons: Optional[list[str]] = Field(default=None, description="Write down all the cons inside a list")
    name: Optional[str] = Field(default=None, description="Write the name of the reviewer")

# Link the model with the structured output schema defined above
structured_model = model.with_structured_output(Review)

# Invoke the model with a product review and parse it into structured output
result = structured_model.invoke("""
I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! 
The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. 
The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. 
What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. 
Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. 
Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? 
The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by Nitish Singh
""")

# Print the final structured result (will be an instance of the Review model)
print(result)
