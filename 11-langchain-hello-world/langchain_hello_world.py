"""
LangChain Hello World
=====================
Demonstrates the three core LangChain primitives:
  1. ChatOpenAI      — the LLM wrapper
  2. PromptTemplate  — parameterised, reusable prompt construction
  3. StrOutputParser — clean text extraction from an LLM response

Runs three progressively richer chains so you can see exactly what
each layer adds.

Requirements:
    pip install langchain>=0.2.0 langchain-openai>=0.1.0

Environment:
    export OPENAI_API_KEY="your-key-here"
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ---------------------------------------------------------------------------
# Step 1: Initialise the LLM
# ---------------------------------------------------------------------------
def get_llm() -> ChatOpenAI:
    """Return a ChatOpenAI instance using the key from the environment."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. "
            "Run: export OPENAI_API_KEY='your-key-here'"
        )
    return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)


# ---------------------------------------------------------------------------
# Step 2: Minimal chain — raw LLM call, no template, no parser
# ---------------------------------------------------------------------------
def demo_raw_llm(llm: ChatOpenAI) -> None:
    """
    Simplest possible LangChain interaction: pass a plain string to the LLM
    and print the AIMessage object that comes back.

    Notice that the response is an AIMessage, not a plain string.
    That's what StrOutputParser solves in the next demo.
    """
    print("\n" + "=" * 60)
    print("DEMO 1: Raw LLM call (no template, no parser)")
    print("=" * 60)

    response = llm.invoke("Say 'Hello, LangChain!' and explain in one sentence what LangChain is.")

    # response is an AIMessage object
    print(f"Type  : {type(response)}")
    print(f"Output: {response.content}")


# ---------------------------------------------------------------------------
# Step 3: PromptTemplate — parameterise the prompt
# ---------------------------------------------------------------------------
def demo_prompt_template(llm: ChatOpenAI) -> None:
    """
    Use ChatPromptTemplate to separate prompt logic from application code.

    The template accepts a {topic} variable. This means the same prompt
    structure can be reused for any topic without copy-pasting strings.
    """
    print("\n" + "=" * 60)
    print("DEMO 2: PromptTemplate — parameterised prompt")
    print("=" * 60)

    # Define the prompt template with a variable placeholder
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a concise technical educator. Answer in 2–3 sentences."),
        ("human", "Explain {topic} to a Python developer who is new to AI."),
    ])

    # Render the prompt by filling in the variable — no LLM call yet
    filled_prompt = prompt.format_messages(topic="vector embeddings")
    print("Rendered prompt (human turn):")
    print(f"  {filled_prompt[-1].content}")

    # Now invoke: prompt → llm
    chain = prompt | llm
    response = chain.invoke({"topic": "vector embeddings"})
    print(f"\nLLM response (AIMessage):")
    print(f"  {response.content}")


# ---------------------------------------------------------------------------
# Step 4: Full chain — PromptTemplate | LLM | StrOutputParser
# ---------------------------------------------------------------------------
def demo_full_chain(llm: ChatOpenAI) -> None:
    """
    Add StrOutputParser to the end of the chain.

    StrOutputParser extracts .content from the AIMessage, returning a plain
    Python string. This is the standard pattern for LangChain chains that
    feed their output into downstream steps or application logic.
    """
    print("\n" + "=" * 60)
    print("DEMO 3: Full chain — PromptTemplate | LLM | StrOutputParser")
    print("=" * 60)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Be concise."),
        ("human", "Give me one practical use case for {technology} in AWS cloud infrastructure."),
    ])

    # The pipe operator composes the three steps into a single Runnable
    chain = prompt | llm | StrOutputParser()

    technologies = ["LangChain", "vector databases", "LLM agents"]

    for tech in technologies:
        # .invoke() runs all three steps end-to-end
        result = chain.invoke({"technology": tech})
        # result is now a plain Python string — ready to use anywhere
        print(f"\n[{tech}]")
        print(f"  {result}")


# ---------------------------------------------------------------------------
# Step 5: Batch invocation — run the same chain for multiple inputs at once
# ---------------------------------------------------------------------------
def demo_batch(llm: ChatOpenAI) -> None:
    """
    Use .batch() to run the chain for multiple inputs efficiently.

    LangChain processes each input independently but in a single call to
    your code — useful for bulk processing tasks.
    """
    print("\n" + "=" * 60)
    print("DEMO 4: Batch invocation — multiple inputs in one .batch() call")
    print("=" * 60)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a technical glossary. Define terms in exactly one sentence."),
        ("human", "Define: {term}"),
    ])

    chain = prompt | llm | StrOutputParser()

    terms = [
        {"term": "RAG (Retrieval-Augmented Generation)"},
        {"term": "LangChain Expression Language (LCEL)"},
        {"term": "embedding model"},
    ]

    results = chain.batch(terms)

    for input_data, output in zip(terms, results):
        print(f"\n{input_data['term']}:")
        print(f"  {output}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("🦜 LangChain Hello World")
    print("Demonstrating: ChatOpenAI | PromptTemplate | StrOutputParser\n")

    llm = get_llm()

    demo_raw_llm(llm)
    demo_prompt_template(llm)
    demo_full_chain(llm)
    demo_batch(llm)

    print("\n" + "=" * 60)
    print("All demos complete.")
    print("Next step: Project 12 — Conversational chatbot with memory.")
    print("=" * 60)


if __name__ == "__main__":
    main()
