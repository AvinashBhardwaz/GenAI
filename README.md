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

## 📐 Embeddings

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
