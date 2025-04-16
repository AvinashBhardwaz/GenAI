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
    template=\"\"\"
    Please summarize the research paper titled "{paper_input}"...
    \"\"\",
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

## 🔄 Structured Output Plan

You're now transitioning toward **structured output** using:

- `JsonOutputParser`
- `PydanticOutputParser`
- `Function calling` (planned)

---

## ✅ Next Step

Start using **structured outputs** to make your application more robust and production-ready.

---

## 💡 Final Thought

> You're on the right path! Learning by doing, combining LangChain + OpenAI + HuggingFace + Streamlit is an ideal modern AI dev stack.
