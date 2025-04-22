from langchain_experimental.text_splitter import SemanticChunker
# What: Importing the experimental semantic-aware chunker.
# Why: Unlike character- or syntax-based splitters, this one uses **semantic similarity** to decide chunk boundaries.
# How: It uses embeddings to compare adjacent sections of text and finds “breakpoints” where semantic drift is high.

from langchain_openai.embeddings import OpenAIEmbeddings
# What: Imports OpenAI's embedding model wrapper.
# Why: SemanticChunker needs a model to compute vector representations of text blocks.
# How: The embeddings help it decide where topic/context naturally shifts in the text.

from dotenv import load_dotenv
# What: Used to load API keys and environment variables from a .env file.
# Why: Required if you're storing your OpenAI API key in a .env file.
# How: Ensures the environment is set up for OpenAI requests.

load_dotenv()
# What: Loads environment variables from the .env file into the runtime.
# Why: So that `OpenAIEmbeddings()` can access your API key without hardcoding it.
# How: You just need to make sure `.env` contains `OPENAI_API_KEY=your-key`.

# Initialize the semantic-aware chunker
text_splitter = SemanticChunker(
    OpenAIEmbeddings(),                          # What: Pass in an embedding model
                                                 # Why: Needed to compute similarity between chunks
    breakpoint_threshold_type="standard_deviation",  # What: Determines how semantic "distance" is evaluated
                                                     # Why: "standard_deviation" uses dynamic thresholds for better generalization
    breakpoint_threshold_amount=3                # What: Sets how sensitive the break detection is
                                                 # Why: Higher values = fewer chunks, only big topic shifts trigger a split
)

sample = """
Farmers were working hard in the fields, preparing the soil and planting seeds for the next season. The sun was bright, and the air smelled of earth and fresh grass. The Indian Premier League (IPL) is the biggest cricket league in the world. People all over the world watch the matches and cheer for their favourite teams.

Terrorism is a big danger to peace and safety. It causes harm to people and creates fear in cities and villages. When such attacks happen, they leave behind pain and sadness. To fight terrorism, we need strong laws, alert security forces, and support from people who care about peace and safety.
"""
# What: Sample text with a few topic shifts: farming, cricket, and terrorism.
# Why: Ideal for semantic chunking, which detects where ideas change meaningfully.
# How: The chunker should ideally separate sports and terrorism into different chunks.

docs = text_splitter.create_documents([sample])
# What: Creates Document objects with semantically consistent chunks.
# Why: Useful for building better RAG pipelines or understanding topic shifts.
# How: Embeds and compares overlapping blocks, splits where meaning changes.

print(len(docs))  # What: Prints number of chunks created
                  # Why: Verifies how sensitive the split threshold was
print(docs)       # What: Prints the resulting Document objects
                  # Why: Lets you inspect the split content
