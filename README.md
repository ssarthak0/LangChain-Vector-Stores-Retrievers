# LangChain Vector Stores & Retrievers

This tutorial introduces LangChain’s abstractions for working with vector stores and retrievers, useful for Retrieval-Augmented Generation (RAG) workflows. We walk through how to use documents, store them in a vector database, retrieve relevant context using similarity search, and build an end-to-end RAG pipeline.

---

## 📦 Installation

```bash
pip install langchain
pip install langchain-chroma
pip install langchain_groq
pip install langchain_huggingface
```

---

## 📄 Documents

LangChain provides a Document abstraction with:

- `page_content`: The main text content.
- `metadata`: A dictionary of associated metadata.

```python
from langchain_core.documents import Document

documents = [
    Document(page_content="Dogs are great companions, known for their loyalty and friendliness.", metadata={"source": "mammal-pets-doc"}),
    Document(page_content="Cats are independent pets that often enjoy their own space.", metadata={"source": "mammal-pets-doc"}),
    Document(page_content="Goldfish are popular pets for beginners, requiring relatively simple care.", metadata={"source": "fish-pets-doc"}),
    Document(page_content="Parrots are intelligent birds capable of mimicking human speech.", metadata={"source": "bird-pets-doc"}),
    Document(page_content="Rabbits are social animals that need plenty of space to hop around.", metadata={"source": "mammal-pets-doc"}),
]
```

---

## 🔐 Environment Setup

```python
import os
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
```

---

## 💬 Initialize LLM

```python
from langchain_groq import ChatGroq

llm = ChatGroq(groq_api_key=groq_api_key, model="Llama3-8b-8192")
```

---

## 🧠 Embeddings and Vector Store

```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents, embedding=embeddings)
```

---

## 🔍 Similarity Search

```python
vectorstore.similarity_search("cat")
await vectorstore.asimilarity_search("cat")
vectorstore.similarity_search_with_score("cat")
```

---

## 🔁 Using Retrievers

LangChain VectorStore objects are not Runnable. To integrate them with chains, you need to use retrievers which are Runnable-compatible.

### Custom Retriever Using RunnableLambda

```python
from langchain_core.runnables import RunnableLambda

retriever = RunnableLambda(vectorstore.similarity_search).bind(k=1)
retriever.batch(["cat", "dog"])
```

### Standard Retriever via `as_retriever()`

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1}
)
retriever.batch(["cat", "dog"])
```

---

## 🔗 RAG Pipeline

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

prompt_template = """
Answer this question using the provided context only.

{question}

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([("human", prompt_template)])

rag_chain = {
    "context": retriever,
    "question": RunnablePassthrough()
} | prompt | llm

response = rag_chain.invoke("tell me about dogs")
print(response.content)
```

---

## ✅ Summary

This notebook demonstrates how to:

- Represent text as documents
- Embed and store them in Chroma vector database
- Perform similarity search
- Wrap retrieval in LangChain retrievers
- Build a simple RAG pipeline with context + question → LLM

> 🧪 Try modifying the document content or changing `k` in retriever settings to experiment with different retrieval results.

Happy building with LangChain! 🚀
