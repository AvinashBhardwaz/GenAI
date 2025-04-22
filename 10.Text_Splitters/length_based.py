from langchain.text_splitter import CharacterTextSplitter
# What: Imports CharacterTextSplitter from LangChain.
# Why: Used to split text/documents into smaller chunks based on character limits.
# How: Allows control over chunk size, overlap, and splitting logic.

from langchain_community.document_loaders import PyPDFLoader
# What: Imports a PDF loader from the community module.
# Why: PyPDFLoader reads PDF files and converts them into LangChain Document objects.
# How: Each page in the PDF becomes a Document, allowing further processing like splitting.

loader = PyPDFLoader('dl-curriculum.pdf')
# What: Creates a loader instance for the specified PDF.
# Why: Required to load the document into LangChain’s workflow.
# How: Pass the path to your PDF file; loader handles the rest.

docs = loader.load()
# What: Loads the PDF document into memory.
# Why: Needed to extract text content from the file.
# How: Returns a list of Document objects (usually one per page).

splitter = CharacterTextSplitter(
    chunk_size=200,     # Max characters per chunk
    chunk_overlap=0,    # No overlap between chunks
    separator=''        # Split at every character (not word/line based)
)
# What: Initializes the text splitter with specific settings.
# Why: Needed to break long documents into smaller chunks for better processing (e.g., in LLMs).
# How: `chunk_size` sets max length per chunk; `chunk_overlap` defines redundancy; `separator` is where to split.

result = splitter.split_documents(docs)
# What: Splits the loaded documents into smaller chunks.
# Why: Long texts need to be split for efficient vectorization and LLM input.
# How: Applies the character-based split logic on all documents.

#print(result)
print(result[0].page_content)
# What: Prints the content of the second chunk (index 1).
# Why: To verify the split results.
# How: Access the `page_content` attribute of the Document object.
