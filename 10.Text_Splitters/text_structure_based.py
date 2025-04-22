from langchain.text_splitter import RecursiveCharacterTextSplitter
# What: Importing RecursiveCharacterTextSplitter from LangChain.
# Why: This splitter is smarter than a simple character splitter—it tries to split at natural boundaries like paragraphs or sentences first.
# How: It recursively uses a list of preferred separators to make clean splits (e.g., `\n\n`, `\n`, `.`, etc.).
text = """
Space exploration has led to incredible scientific discoveries. From landing on the Moon to exploring Mars, humanity continues to push the boundaries of what’s possible beyond our planet.

These missions have not only expanded our knowledge of the universe but have also contributed to advancements in technology here on Earth. Satellite communications, GPS, and even certain medical imaging techniques trace their roots back to innovations driven by space programs.
"""
# What: A long multi-paragraph text string about space exploration.
# Why: This is the raw input we want to break into chunks that are easier to process in an LLM.
# How: Will be passed to the text splitter.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,     # Max number of characters in each chunk
    chunk_overlap=0,    # No overlap between chunks
)
# What: Initializes a recursive splitter object.
# Why: This splitter tries to maintain semantic meaning by avoiding mid-sentence or mid-word splits.
# How: It works through a list of separators, splitting at the largest meaningful boundary possible within the chunk size.
chunks = splitter.split_text(text)
# What: Splits the input text into chunks.
# Why: Useful for LLM input, where max token limits or context size requires smaller text units.
# How: Applies the recursive splitting strategy based on defined chunk size and overlap.
print(len(chunks))
print(chunks)
# What: Prints the number of chunks and their content.
# Why: To verify that the text was split correctly and meaningfully.
# How: Shows the output chunks as a list of strings.
