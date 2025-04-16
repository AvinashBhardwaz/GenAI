# Import necessary modules
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt

# Load environment variables (e.g., OpenAI API Key from .env)
load_dotenv()

# Initialize OpenAI chat model
model = ChatOpenAI()

# Set Streamlit page title
st.header('🧠 Research Paper Summarization Tool')

# Dropdown to select the research paper
paper_input = st.selectbox(
    "📄 Select Research Paper Name",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis"
    ]
)

# Dropdown to select explanation style
style_input = st.selectbox(
    "🎨 Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

# Dropdown to select explanation length
length_input = st.selectbox(
    "📏 Select Explanation Length",
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
)

# Load prompt template from file
template = load_prompt('template.json')

# Button to trigger summarization
if st.button('🔍 Summarize'):
    chain = template | model  # Combine prompt and model into a chain
    result = chain.invoke({   # Pass user selections as input to the prompt
        'paper_input': paper_input,
        'style_input': style_input,
        'length_input': length_input
    })
    # Display the result
    st.subheader("📘 Summary")
    st.write(result.content)
