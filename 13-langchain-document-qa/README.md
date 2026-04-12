# 🦜 LangChain Document QA — AI Projects

> **Load any text or PDF document and ask natural-language questions about it — using LangChain's document loaders, text splitters, and RetrievalQA chain.**

## 📖 What This Project Does

Most real AI applications don't run on general knowledge — they run on *your* documents. This project shows how to build a document question-answering (QA) system using LangChain. You load a document (text or PDF), split it into manageable chunks, embed those chunks into a FAISS vector store, and wire up a `RetrievalQA` chain that fetches the most relevant chunks and passes them to an LLM to generate a grounded answer.

This pattern — often called Retrieval-Augmented Generation (RAG) — is the foundation of enterprise AI assistants, internal knowledge bases, and document chatbots. Project 16 will go deeper with FAISS and OpenAI embeddings at scale, but this project gives you the core mechanics in their simplest form.

The bundled sample document covers AWS cloud services, so you can ask questions like *"What is EC2?"* or *"How does S3 handle durability?"* and get precise, document-grounded answers with source attribution.

## 🧠 Concepts Covered

- **Document Loaders** (`TextLoader`, `PyPDFLoader`) — ingesting raw files into LangChain `Document` objects
- **Text Splitting** (`RecursiveCharacterTextSplitter`) — chunking long documents into LLM-friendly pieces with controlled overlap
- **FAISS Vector Store** — storing and retrieving document embeddings by semantic similarity
- **OpenAI Embeddings** (`text-embedding-3-small`) — converting text chunks into dense numerical vectors
- **RetrievalQA Chain** — combining a retriever and an LLM into a grounded question-answering pipeline
- **Custom PromptTemplate** — controlling exactly how context and questions are presented to the LLM
- **Source Attribution** (`return_source_documents=True`) — knowing which chunks backed each answer

## 🚀 How to Run

### Prerequisites
- Python 3.9+
- An OpenAI API key

### Setup

```bash
cd 13-langchain-document-qa
pip install -r requirements.txt
```

Set your API key:
```bash
export OPENAI_API_KEY="your-key-here"
```

### Run the script
```bash
python langchain_document_qa.py
```

The script will:
1. Create a `sample_document.txt` about AWS cloud services
2. Load, split, and embed the document into FAISS
3. Answer five preset questions automatically
4. Drop into an interactive Q&A mode — type any question, or `quit` to exit

### Use your own document
Swap the `TextLoader` path for any `.txt` file, or replace it with `PyPDFLoader("your_file.pdf")` for PDFs.

### Or open the notebook
```bash
jupyter notebook langchain_document_qa.ipynb
```

## 📚 Key Takeaways

- **RAG in 4 steps**: Load → Split → Embed → Retrieve + Generate. This project makes each step explicit and observable.
- **Chunk size matters**: `chunk_size=500` with `chunk_overlap=50` is a solid starting point — larger chunks provide more context but consume more tokens per call.
- **FAISS is in-memory**: Fast and zero-config for prototyping. Project 23 (`langchain-chroma-vectordb`) covers persisting a vector DB across sessions.
- **`return_source_documents=True`**: Always know which chunks the LLM used — critical for debugging hallucinations and building user trust.
- **Temperature=0**: For factual document QA, keep it at zero to avoid the LLM drifting beyond the source material.

---
*Part of the [AI Projects](https://github.com/naveenramasamy11/ai-projects) series.*
