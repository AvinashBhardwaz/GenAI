# Import PyPDFLoader to load and process PDF files
from langchain_community.document_loaders import PyPDFLoader

# What: Create a loader to read a single PDF file
# Why: To extract text and metadata from the given PDF file for processing
# How: Initialize PyPDFLoader with the file path to the PDF
loader = PyPDFLoader('dl-curriculum.pdf')

# What: Load all pages of the PDF
# Why: To get the document split into individual pages (each page becomes a Document object)
# How: .load() reads the entire PDF and returns a list of documents (one per page)
docs = loader.load()

# What: Print how many pages (documents) were loaded
# Why: To check how many pages the loader was able to process
# How: Use len() on the list of documents
print(len(docs))

# What: Print the actual content of the first page
# Why: To verify that the text was extracted correctly from the page
# How: Access the 'page_content' attribute of the first document
print(docs[0].page_content)

# What: Print metadata for the second page
# Why: Metadata contains useful info like page number and file source
# How: Access the 'metadata' attribute of the second document
print(docs[1].metadata)
