from langchain_community.document_loaders import CSVLoader  # ✅ Importing the CSVLoader to load CSV files into LangChain documents

# ✅ Create a CSVLoader instance by providing the path to the CSV file
# 🔍 WHAT: CSVLoader reads CSV files and turns each row into a document
# ❓ WHY: To allow language models to process and understand tabular data as documents
# ⚙️ HOW: Internally, each row becomes a LangChain `Document` object with content + metadata
loader = CSVLoader(file_path='Social_Network_Ads.csv')

# ✅ Load the documents from the CSV file
# This returns a list of Document objects
docs = loader.load()

# ✅ Print the number of documents (i.e., rows in the CSV excluding headers)
print(len(docs))

# ✅ Print the second document (index 1)
print(docs[2])
