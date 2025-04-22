from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
# What: Importing LangChain's intelligent text splitter and language enum.
# Why: We're using a code-aware splitter that understands Python syntax structure (functions, classes, etc.).
# How: `Language.PYTHON` enables structured chunking at logical Python blocks like functions or methods.

text = """
class Student:
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade  # Grade is a float (like 8.5 or 9.2)

    def get_details(self):
        return self.name"

    def is_passing(self):
        return self.grade >= 6.0


# Example usage
student1 = Student("Aarav", 20, 8.2)
print(student1.get_details())

if student1.is_passing():
    print("The student is passing.")
else:
    print("The student is not passing.")
"""
# What: A small Python class with student attributes and behavior.
# Why: Great candidate for testing the Python-aware splitter’s ability to break at clean boundaries like class methods or blocks.
# How: Each function or logical group of statements can become a separate chunk.

# Initialize the code-aware splitter
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,  # What: Informs the splitter to parse this as Python code
                               # Why: Allows it to split intelligently at function/class level
    chunk_size=300,            # What: Set a 300 character limit per chunk
                               # Why: Keeps chunks small enough for downstream tasks like code summarization or LLM input
    chunk_overlap=0            # What: No overlap between consecutive chunks
                               # Why: Ensures each chunk is distinct
)

# Perform the splitting
chunks = splitter.split_text(text)
# What: Splits the code into logical, semantically meaningful chunks.
# Why: Helps improve performance and reasoning of downstream tools (like LLMs).
# How: Uses recursive rules based on Python syntax and chunk length.

print(len(chunks))     # What: Prints the number of chunks created
                       # Why: Useful for confirming how granular the split was
print(chunks[1])       # What: Shows the content of the second chunk
                       # Why: Lets you inspect how the Python-aware logic handled the splitting
