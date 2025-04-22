from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
# What: Importing both the advanced text splitter and the supported language enum.
# Why: `Language.MARKDOWN` allows the splitter to intelligently parse markdown structure (e.g., headers, lists).
# How: The splitter will prioritize splitting at markdown boundaries like headings and list items.

text = """
# Project Name: Smart Student Tracker

A simple Python-based project to manage and track student data, including their grades, age, and academic status.


## Features

- Add new students with relevant info
- View student details
- Check if a student is passing
- Easily extendable class-based design


## 🛠 Tech Stack

- Python 3.10+
- No external dependencies


## Getting Started

1. Clone the repo  
   ```bash
   git clone https://github.com/your-username/student-tracker.git
"""
# What: Sample markdown documentation for a student tracking project.
# Why: Great test case to see how markdown-aware splitting works.
# How: This multi-section markdown content will be broken along logical boundaries (e.g., between headings or list blocks).

# Initialize the splitter using the markdown-aware preset
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,  # What: Tells the splitter to treat input as markdown
                                 # Why: Helps split cleanly around headings, bullets, etc.
    chunk_size=200,              # What: Max 200 characters per chunk
                                 # Why: Mimics LLM input size limits
    chunk_overlap=0              # What: No overlapping between chunks
                                 # Why: Keeps chunks unique and concise
)

# Perform the actual split
chunks = splitter.split_text(text)
# What: Splits the text into chunks using markdown-aware logic
# Why: Prepares input for downstream processing like summarization, QA, etc.
# How: Internally uses a hierarchy of separators optimized for markdown

print(len(chunks))     # What: Shows how many chunks were created
                       # Why: Useful to confirm split behavior
print(chunks[0])       # What: Prints the first chunk
                       # Why: Helps visually inspect the result
