# 🦜 LangChain Hello World — LangChain AI Projects

> **Your first LangChain chain: wire a PromptTemplate, an LLM, and an output parser together to understand how LangChain's core building blocks compose.**

## 📖 What This Project Does

LangChain's power comes from the ability to chain together prompts, language models, and output parsers into reusable, composable pipelines. Before you can build RAG systems or autonomous agents, you need a solid grasp of these three primitives — and that's exactly what this project teaches.

This project walks through building three progressively richer chains. The first is a minimal "Hello, LangChain!" call that sends a prompt to an OpenAI chat model and prints the raw response. The second introduces `PromptTemplate` to parameterise prompts and separate concerns — your prompt logic lives in the template, not scattered through your code. The third adds a `StrOutputParser` to cleanly extract the text content from the LLM response object, producing a plain Python string ready for downstream use.

By the end you'll understand how `PromptTemplate | LLM | OutputParser` forms the backbone of virtually every LangChain application, from simple chatbots to multi-step agents. Every advanced LangChain pattern builds on exactly this foundation.

## 🧠 Concepts Covered

- `ChatOpenAI` — initialising an LLM via `langchain_openai`
- `PromptTemplate` and `ChatPromptTemplate` — parameterised, reusable prompt construction
- `StrOutputParser` — extracting plain text from an LLM response
- The pipe `|` operator (LCEL) — composing chain steps declaratively
- Invoking a chain with `.invoke()` and inspecting the response

## 🚀 How to Run

### Prerequisites
- Python 3.9+
- An OpenAI API key

### Setup

```bash
cd 11-langchain-hello-world
pip install -r requirements.txt
```

Set your API key:
```bash
export OPENAI_API_KEY="your-key-here"
```

### Run the script
```bash
python langchain_hello_world.py
```

### Or open the notebook
```bash
jupyter notebook langchain_hello_world.ipynb
```

## 📚 Key Takeaways

- A LangChain chain is built by piping components together: `prompt | llm | parser`
- `PromptTemplate` keeps prompt logic separate from application code, making it easy to reuse and test prompts independently
- `StrOutputParser` unwraps the `AIMessage` object so downstream code works with plain strings — essential for building larger pipelines

---
*Part of the [AI Projects](https://github.com/naveenramasamy11/ai-projects) series.*
