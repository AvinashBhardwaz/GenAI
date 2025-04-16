# Import the ChatGoogleGenerativeAI class to interact with Google's Gemini models
from langchain_google_genai import ChatGoogleGenerativeAI

# Import load_dotenv to load environment variables from a .env file
from dotenv import load_dotenv

# Load environment variables (e.g., Google API key) from the .env file into the environment
load_dotenv()

# Initialize the Google Generative AI model using the 'gemini-1.5-pro' version
model = ChatGoogleGenerativeAI(model='gemini-1.5-pro')

# Send a prompt to the Gemini model asking for the capital of India
result = model.invoke('What is the capital of India')

# Print only the content (text) portion of the model's response
print(result.content)
