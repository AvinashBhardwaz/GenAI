# Import the PromptTemplate class to define reusable prompt templates
from langchain_core.prompts import PromptTemplate

# Define a prompt template for summarizing a research paper
template = PromptTemplate(
    template="""
Please summarize the research paper titled "{paper_input}" with the following specifications:
Explanation Style: {style_input}  
Explanation Length: {length_input}  

1. Mathematical Details:  
   - Include relevant mathematical equations if present in the paper.  
   - Explain the mathematical concepts using simple, intuitive code snippets where applicable.  

2. Analogies:  
   - Use relatable analogies to simplify complex ideas.  

If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.  

Ensure the summary is clear, accurate, and aligned with the provided style and length.
""",
    
    # Define the input variables that will be dynamically filled at runtime
    input_variables=['paper_input', 'style_input', 'length_input'],
    
    # Ensure that the template and input variables match correctly
    validate_template=True
)

# Save the template to a JSON file for future reuse or sharing
template.save('template.json')
