# 🦜 LangChain Chatbot with Conversation Memory — AI Projects

> **Build a stateful conversational chatbot that actually remembers what you said earlier — using `ConversationBufferMemory` and `ConversationChain`.**

## 📖 What This Project Does

Most LLM API calls are stateless — send a message, get a reply, and the model immediately forgets the exchange. Real chatbots need to carry context forward: if you tell the bot your name in turn 1, it should still know it in turn 10. LangChain solves this with **memory objects** that store and inject conversation history automatically.

This project demonstrates the simplest and most commonly used memory primitive: `ConversationBufferMemory`. It keeps the **full verbatim history** of every human/AI turn in RAM and automatically injects it into the prompt on every call — so the LLM always sees the complete conversation context.

We wire everything together using `ConversationChain`, which handles the read-from-memory → format-prompt → call-LLM → write-to-memory loop on every `.predict()` call. You also learn how to write a custom `PromptTemplate` with `{history}` and `{input}` placeholders, giving you full control over the system persona and conversation format.

## 🧠 Concepts Covered

- `ConversationBufferMemory` — storing and retrieving full chat history
- `ConversationChain` — the standard LangChain chain for stateful conversations
- Custom `PromptTemplate` with `{history}` and `{input}` variables
- How LangChain's memory read/write cycle works under the hood
- Interactive REPL chat loop vs. scripted demo mode

## 🚀 How to Run

### Prerequisites
- Python 3.9+
- An OpenAI API key

### Setup

```bash
cd 12-langchain-chatbot-memory
pip install -r requirements.txt
```

Set your API key:
```bash
export OPENAI_API_KEY="your-key-here"
```

### Run the script (scripted demo by default)
```bash
python langchain_chatbot_memory.py
```

Switch to interactive mode by editing line `demo_mode = True` → `demo_mode = False` in `main()`.

### Or open the notebook
```bash
jupyter notebook langchain_chatbot_memory.ipynb
```

## 📚 Key Takeaways

- `ConversationBufferMemory` is the simplest LangChain memory — it stores every turn verbatim. For long conversations, consider `ConversationSummaryMemory` (summarises older turns) or `ConversationBufferWindowMemory` (keeps only the last N turns).
- `ConversationChain.predict(input=...)` handles the entire memory read → prompt format → LLM call → memory write cycle for you.
- The `{history}` variable in the prompt template is populated automatically by the memory object — you never have to manage it manually.
- Inspecting `chain.memory.buffer` lets you see exactly what context the model receives on the next turn — essential for debugging.

---
*Part of the [AI Projects](https://github.com/naveenramasamy11/ai-projects) series.*
