"""
LangChain Chatbot with Conversation Memory
===========================================
Demonstrates how to build a stateful conversational chatbot using
ConversationBufferMemory and ConversationChain with LangChain v0.2+.

Author: AI Projects Series
"""

import os
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_core.prompts import PromptTemplate


# ---------------------------------------------------------------------------
# Step 1: Configure the LLM
# ---------------------------------------------------------------------------
def create_llm(model_name: str = "gpt-3.5-turbo", temperature: float = 0.7) -> ChatOpenAI:
    """
    Initialise the ChatOpenAI model.
    Reads OPENAI_API_KEY from the environment.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set.\n"
            "Export it first:  export OPENAI_API_KEY='your-key-here'"
        )
    return ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key)


# ---------------------------------------------------------------------------
# Step 2: Set up conversation memory
# ---------------------------------------------------------------------------
def create_memory() -> ConversationBufferMemory:
    """
    ConversationBufferMemory stores the entire chat history in-memory.
    The 'history' key is injected into the prompt template automatically.
    """
    return ConversationBufferMemory(
        memory_key="history",
        return_messages=False,  # return as plain text, not a list of Message objects
    )


# ---------------------------------------------------------------------------
# Step 3: Build a custom prompt template
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_TEMPLATE = """\
You are a helpful, friendly AI assistant. You remember everything that has
been said in this conversation and use that context to give accurate,
relevant answers. If you don't know something, say so clearly.

Conversation history:
{history}
Human: {input}
Assistant:"""


def create_prompt() -> PromptTemplate:
    """Return a PromptTemplate that accepts {history} and {input}."""
    return PromptTemplate(
        input_variables=["history", "input"],
        template=SYSTEM_PROMPT_TEMPLATE,
    )


# ---------------------------------------------------------------------------
# Step 4: Assemble the ConversationChain
# ---------------------------------------------------------------------------
def build_chain(llm: ChatOpenAI, memory: ConversationBufferMemory,
                prompt: PromptTemplate) -> ConversationChain:
    """
    ConversationChain wires together the LLM, the memory, and the prompt.
    On every call it:
      1. Loads existing chat history from memory
      2. Formats the prompt with {history} + {input}
      3. Calls the LLM
      4. Saves the human/AI exchange back to memory
    """
    return ConversationChain(
        llm=llm,
        memory=memory,
        prompt=prompt,
        verbose=False,  # set True to see the full formatted prompt each turn
    )


# ---------------------------------------------------------------------------
# Step 5: Interactive chat loop
# ---------------------------------------------------------------------------
def chat_loop(chain: ConversationChain) -> None:
    """
    Run an interactive REPL until the user types 'quit' or 'exit'.
    Demonstrates that the chatbot remembers context across turns.
    """
    print("\n" + "=" * 60)
    print(" LangChain Memory Chatbot — type 'quit' to exit")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Session ended by user]")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        response = chain.predict(input=user_input)
        print(f"\nAssistant: {response}\n")


# ---------------------------------------------------------------------------
# Step 6: Demo run (non-interactive, useful for testing)
# ---------------------------------------------------------------------------
def demo_run(chain: ConversationChain) -> None:
    """
    Send a scripted sequence of messages to show memory in action.
    The assistant should remember facts from earlier turns.
    """
    exchanges = [
        "Hi! My name is Naveen and I'm an AWS cloud consultant.",
        "What services do I typically work with as an AWS consultant?",
        "Can you remind me what my job title is?",          # tests memory
        "What's a good LangChain project for someone in my field?",
    ]

    print("\n" + "=" * 60)
    print(" LangChain Memory Demo — scripted conversation")
    print("=" * 60)

    for user_msg in exchanges:
        print(f"\nYou:       {user_msg}")
        response = chain.predict(input=user_msg)
        print(f"Assistant: {response}")

    # Show what the memory buffer contains at the end
    print("\n" + "-" * 60)
    print("Memory buffer contents:")
    print("-" * 60)
    print(chain.memory.buffer)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """
    By default, run the scripted demo.
    Change demo=False to start the interactive chat loop.
    """
    demo_mode = True  # set False for interactive REPL

    llm    = create_llm()
    memory = create_memory()
    prompt = create_prompt()
    chain  = build_chain(llm, memory, prompt)

    if demo_mode:
        demo_run(chain)
    else:
        chat_loop(chain)


if __name__ == "__main__":
    main()
