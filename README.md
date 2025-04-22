# LangChain Learning Journey 🚀

This README captures the learning path and experiments you've conducted using LangChain, HuggingFace, OpenAI, and Streamlit.

---

## ✅ Environment Setup

- Used `.env` for managing API keys and environment variables.
- Cached HuggingFace models locally using `HF_HOME`.

---

## 🧠 Language Models

### HuggingFace LLMs
```python
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

llm = HuggingFaceEndpoint(repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", task="text-generation")
model = ChatHuggingFace(llm=llm)
result = model.invoke("What is the capital of India")
```

### HuggingFacePipeline (local caching)
```python
from langchain_huggingface import HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs=dict(temperature=0.5, max_new_tokens=100)
)
```

---

## 🖐️ Embeddings

### OpenAI Embeddings
```python
from langchain_openai import OpenAIEmbeddings

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)
result = embedding.embed_query("Delhi is the capital of India")
```

### Similarity with OpenAI Embeddings
```python
from sklearn.metrics.pairwise import cosine_similarity

documents = [...]
query = 'tell me about bumrah'
doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)
scores = cosine_similarity([query_embedding], doc_embeddings)[0]
```

### HuggingFace Embeddings
```python
from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vector = embedding.embed_documents(documents)
```

---

## 💬 Prompt Templates

### ChatPromptTemplate with `MessagesPlaceholder`
```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_template = ChatPromptTemplate([
    ('system','You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{query}')
])
```

#### Why use `MessagesPlaceholder`?
- Keeps full context of the conversation.
- Enables dynamic, multi-turn chat applications.

---

## 🤖 Conversational Chat with Memory
```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

chat_history = [SystemMessage(content='You are a helpful assistant')]

while True:
    ...
```

---

## 📄 Structured Prompt Template

### For Research Summary Tool
```python
from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    template="""
    Please summarize the research paper titled "{paper_input}"...
    """,
    input_variables=['paper_input', 'style_input','length_input']
)
```

---

## 🖼️ Streamlit App: Research Tool
```python
import streamlit as st

st.header('Research Tool')
paper_input = st.selectbox(...)
style_input = st.selectbox(...)
length_input = st.selectbox(...)
...
```

---

## 🔄 Structured Output using Pydantic + TinyLlama

```python
from dotenv import load_dotenv
from typing import Optional, Literal
from pydantic import BaseModel, Field
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

class Review(BaseModel):
    key_themes: list[str] = Field(description="Write down all the key themes discussed in the review in a list")
    summary: str = Field(description="A brief summary of the review")
    sentiment: Literal["pos", "neg"] = Field(description="Return sentiment of the review either negative or positive")
    pros: Optional[list[str]] = Field(default=None, description="Write down all the pros inside a list")
    cons: Optional[list[str]] = Field(default=None, description="Write down all the cons inside a list")
    name: Optional[str] = Field(default=None, description="Write the name of the reviewer")

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("Your product review here")
print(result)
```

---

## 🧪 Output Example
```json
{
  "key_themes": ["camera", "battery", "performance", "One UI", "S-Pen"],
  "summary": "A high-performance phone with a stellar camera and fast charging, but slightly bulky and pricey.",
  "sentiment": "pos",
  "pros": ["200MP camera", "Snapdragon 8 Gen 3", "Fast charging", "S-Pen support"],
  "cons": ["Bulky design", "Bloatware", "Expensive"],
  "name": "Avinash Bhardwaz"
}
```

---

## 📦 Output Parser

### Extract structured fields from plain model output
```python
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Literal, Optional

class Review(BaseModel):
    key_themes: List[str] = Field(description="List of major discussion points")
    summary: str = Field(description="Brief review summary")
    sentiment: Literal["pos", "neg"] = Field(description="Sentiment of the review")
    pros: Optional[List[str]] = Field(default=None, description="List of pros")
    cons: Optional[List[str]] = Field(default=None, description="List of cons")
    name: Optional[str] = Field(default=None, description="Name of the reviewer")

parser = PydanticOutputParser(pydantic_object=Review)

format_instructions = parser.get_format_instructions()

prompt = f"""
Review this: "This phone has a great camera and battery life, but it's bulky."

Format your output as follows:
{format_instructions}
"""

output = model.invoke(prompt)
parsed = parser.parse(output)
print(parsed)
```

---

## ⚖️ Chains and Branching Logic

### Conditional Chain with Sentiment Classifier and Response Generator
```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatOpenAI()
parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into postive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction': parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x: x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = classifier_chain | branch_chain

print(chain.invoke({'feedback': 'This is a beautiful phone'}))
```

This demonstrates:
- **Classification + branching** using `RunnableBranch`
- Multiple prompt templates + multiple models
- Clear execution graph with conditional logic

## Runnable Components in LangChain (with Nakli Examples)

### What Are Runnables?
**Runnables** are modular, chainable components in LangChain that follow a standard interface. They define a single method: `invoke(input_data)`. This enables plug-and-play architecture.

### Why Do Runnables Exist?
- To provide a **standardized way** to compose and connect AI components (prompts, models, parsers, etc.).
- To make pipeline creation more **flexible**, **reusable**, and **composable**.
- Allow easy debugging, swapping, and branching of logic like Lego blocks.

---

### Nakli Runnable Implementations
Below are the simplified versions ("Nakli" = fake/mimic) of LangChain-style Runnables:

#### `NakliPromptTemplate`
- Mimics prompt formatting.
- `invoke(input_dict)` fills template with data.

#### `NakliLLM`
- Mimics a language model.
- `invoke(prompt)` returns a random dummy response.

#### `NakliStrOutputParser`
- Mimics a parser.
- `invoke(input_dict)` just returns the 'response' value.

#### `RunnableConnector`
- Mimics chaining.
- `invoke(input_dict)` runs a list of Runnables in order.

#### Sample Chain:
```python
chain = RunnableConnector([
    NakliPromptTemplate(...),
    NakliLLM(),
    NakliStrOutputParser()
])
```

---

### Real LangChain Examples

#### `RunnableSequence`
- What: A sequence of steps (like Prompt -> Model -> Parser).
- Why: To build modular chains.
- How: `RunnableSequence(prompt, model, parser)`

#### `RunnableParallel`
- What: Runs multiple Runnables at the same time.
- Why: To get different outputs from same input (e.g., joke + explanation).
- How: Uses a dictionary of Runnables.

#### `RunnablePassthrough`
- What: A pass-through step.
- Why: Used in branches when no change is needed.
- How: Just returns input.

#### `RunnableLambda`
- What: Wraps a custom Python function into a Runnable.
- Why: Useful for logic like word count, filtering.
- How: `RunnableLambda(func)`

#### `RunnableBranch`
- What: Executes different logic paths based on conditions.
- Why: For conditional logic like summarizing only long texts.
- How: `RunnableBranch((condition, step_if_true), default_step)`

---
# LangChain Runnable Playground

This project explores the concept of Runnables in LangChain, demonstrating how to build flexible and modular chains using mock components like LLMs, prompt templates, and output parsers.

## Sections

### 1. Runnable Components
We re-implemented basic LangChain concepts from scratch using custom classes such as `NakliLLM`, `NakliPromptTemplate`, `NakliStrOutputParser`, and `RunnableConnector`. Each class follows the Runnable protocol by implementing the `invoke()` method. This approach helps understand LangChain's functional programming style.

#### Custom Runnable Components
- `NakliLLM`: Mock language model that randomly selects a pre-defined response.
- `NakliPromptTemplate`: Formats input based on a template string.
- `NakliStrOutputParser`: Extracts the final string from the LLM output.
- `RunnableConnector`: Chains multiple `Runnable` components in sequence.

#### Example
```python
chain = RunnableConnector([template, llm, parser])
result = chain.invoke({'length': 'short', 'topic': 'India'})
```

### 2. LangChain Native Runnables
We then moved to native LangChain classes and structure using `RunnableSequence`, `RunnableParallel`, `RunnablePassthrough`, `RunnableLambda`, and `RunnableBranch`.

#### Examples Covered:
- **RunnableSequence**: Chain components in a sequence (prompt → model → parser)
- **RunnableParallel**: Run multiple branches in parallel (e.g., response + metadata)
- **RunnablePassthrough**: Pass input data without modification (used as an identity function)
- **RunnableLambda**: Add logic with Python functions (e.g., word count)
- **RunnableBranch**: Apply logic based on conditions (e.g., summarize if text is long)

```python
# Example RunnableBranch
branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 300, summarizer_chain),
    RunnablePassthrough()
)
```

### 3. Document Loaders
We explored different document loaders for real-world data ingestion:
- `TextLoader`: Load `.txt` files.
- `CSVLoader`: Load `.csv` files into LangChain documents.
- `PyPDFLoader`: Load PDF files, including multi-page books.
- `DirectoryLoader`: Load all matching files from a directory.
- `WebBaseLoader`: Scrape and load web content.

```python
loader = WebBaseLoader("https://example.com")
docs = loader.load()
```

### 4. Putting It All Together
We demonstrated Q&A over webpage content using:
- `WebBaseLoader` to scrape a product page.
- A `PromptTemplate` to inject the question and content.
- `ChatOpenAI` model to generate the answer.
- `StrOutputParser` to return plain text.

```python
chain = prompt | model | parser
result = chain.invoke({
  'question': 'What is the product?',
  'text': docs[0].page_content
})
```
## 📄 LangChain Experiments

This repo contains my experiments using LangChain with different data sources, models, and chains. Each section demonstrates how to load and process different document types and how to chain models and tools.

---

### 🧱 Runnable Components

LangChain provides a modular approach using `Runnable` classes:
- **RunnableSequence**: A pipeline of chained operations.
- **RunnableParallel**: Executes branches of logic in parallel.
- **RunnablePassthrough**: Passes input directly to output (no transformation).
- **RunnableLambda**: Wraps a Python function for inline transformations.
- **RunnableBranch**: Executes conditional logic like if-else.

🔍 Each component helps structure custom workflows for LLM processing.

---

### 🃏 Prompt + Model + Output Parsing
```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)
model = ChatOpenAI()
parser = StrOutputParser()

chain = prompt | model | parser
```

---

### 🔄 Sequential + Parallel + Branching Chains
```python
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch

# Sequential chain
joke_chain = RunnableSequence(prompt, model, parser)

# Parallel output (get joke and word count)
parallel = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(lambda x: len(x.split()))
})

final_chain = RunnableSequence(joke_chain, parallel)
result = final_chain.invoke({"topic": "AI"})
```

---

### 🗂 Document Loaders
```python
# CSV
from langchain_community.document_loaders import CSVLoader
loader = CSVLoader(file_path='Social_Network_Ads.csv')
docs = loader.load()

# Text
from langchain_community.document_loaders import TextLoader
loader = TextLoader('cricket.txt')
docs = loader.load()

# PDF
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader('dl-curriculum.pdf')
docs = loader.load()

# Directory of PDFs
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
loader = DirectoryLoader(path='books', glob='*.pdf', loader_cls=PyPDFLoader)
docs = loader.lazy_load()

# Web
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://example.com")
docs = loader.load()
```

---

### ✂️ Text Splitters
#### 1. CharacterTextSplitter
```python
from langchain.text_splitter import CharacterTextSplitter
splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
chunks = splitter.split_documents(docs)
```

#### 2. RecursiveCharacterTextSplitter
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
chunks = splitter.split_text(text)
```

#### 3. RecursiveCharacterTextSplitter with Markdown
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN, chunk_size=200, chunk_overlap=0
)
chunks = splitter.split_text(markdown_text)
```

#### 4. RecursiveCharacterTextSplitter with Python Code
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=300, chunk_overlap=0
)
chunks = splitter.split_text(python_code)
```

#### 5. SemanticChunker (Experimental)
```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
text_splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=3
)
docs = text_splitter.create_documents([sample_text])
```

---

### 🧪 Example Use Case: Summarize Web Text
```python
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

loader = WebBaseLoader("https://www.flipkart.com/apple-macbook-air...")
docs = loader.load()

prompt = PromptTemplate(
    template='Answer the following question\n{question} from the following text:\n{text}',
    input_variables=['question','text']
)
model = ChatOpenAI()
parser = StrOutputParser()
chain = prompt | model | parser
print(chain.invoke({"question": "What product is being described?", "text": docs[0].page_content}))
```

---

### 🔧 Setup
```bash
pip install -r requirements.txt
```
Your `.env` file should contain:
```
OPENAI_API_KEY=your-key-here
```

---

More sections coming soon: vector stores, memory, agents, RAG!


---





---