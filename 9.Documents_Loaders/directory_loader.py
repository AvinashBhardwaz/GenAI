# Import DirectoryLoader to load multiple files from a directory
# Import PyPDFLoader to handle reading PDF files
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# What: Creates a loader to read all PDF files in the 'books' directory
# Why: Useful when you want to batch process documents from a folder
# How: Uses DirectoryLoader with a specific loader (PyPDFLoader) and glob pattern to match .pdf files
loader = DirectoryLoader(
    path='books',              # Path to the folder containing PDF files
    glob='*.pdf',              # Match only files ending in .pdf
    loader_cls=PyPDFLoader     # Use PyPDFLoader to read the PDFs
)

# What: Loads PDF documents lazily (one by one instead of all at once)
# Why: Helps with memory efficiency when loading many or large files
# How: Returns a generator object instead of a list
docs = loader.lazy_load()

# What: Iterate through the loaded PDF documents
# Why: To access each document and perform operations like viewing metadata
# How: `docs` is a generator; we loop through it and print metadata for each document
for document in docs:
    print(document.metadata)  # Print metadata such as file path, page number, etc.