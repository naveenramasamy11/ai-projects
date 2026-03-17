"""
Text Summarizer — Extractive Summarization using Word Frequency
================================================================
This project demonstrates how to build a simple extractive text summarizer
using natural language processing (NLP) techniques. Instead of generating
new sentences, it identifies and extracts the most important sentences
from the original text based on word frequency scores.

Key Concepts:
- Tokenization: Splitting text into words and sentences
- Stop words removal: Filtering out common words (the, is, at, ...) that carry little meaning
- Word frequency scoring: Ranking words by how often they appear
- Sentence scoring: Scoring each sentence based on the words it contains
- Extractive summarization: Picking the top-scoring sentences as the summary

Usage:
    python text_summarizer.py
"""

import re
import heapq
import nltk
from collections import defaultdict

# Download required NLTK data files (runs once, cached afterwards)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize


# ---------------------------------------------------------------------------
# Core helper functions
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Remove special characters and extra whitespace from text.

    Parameters
    ----------
    text : str
        Raw input text.

    Returns
    -------
    str
        Cleaned text with only letters, digits, and spaces.
    """
    # Replace anything that is not a letter, digit, or whitespace with a space
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    # Collapse multiple spaces into one
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def compute_word_frequencies(text: str) -> dict:
    """
    Compute the normalized frequency of each meaningful word in the text.

    Stop words (e.g. 'the', 'is', 'in') are excluded because they appear
    very often but carry little meaning.

    Parameters
    ----------
    text : str
        Cleaned input text.

    Returns
    -------
    dict
        Mapping of word -> normalized frequency (0.0 – 1.0).
    """
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text.lower())

    # Count occurrences of each meaningful word
    word_freq: dict = defaultdict(int)
    for word in words:
        if word.isalpha() and word not in stop_words:
            word_freq[word] += 1

    # Normalize by dividing by the maximum frequency
    if word_freq:
        max_freq = max(word_freq.values())
        for word in word_freq:
            word_freq[word] /= max_freq

    return dict(word_freq)


def score_sentences(sentences: list, word_freq: dict) -> dict:
    """
    Score each sentence by summing the frequencies of the words it contains.

    Longer sentences would naturally score higher, so we cap the score at
    30 words to avoid bias towards overly long sentences.

    Parameters
    ----------
    sentences : list of str
        List of sentences from the original text.
    word_freq : dict
        Normalized word frequency dictionary.

    Returns
    -------
    dict
        Mapping of sentence -> cumulative word-frequency score.
    """
    sentence_scores: dict = {}
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        # Only score sentences that are neither too short nor too long
        word_count = len([w for w in words if w.isalpha()])
        if 5 <= word_count <= 30:
            score = sum(word_freq.get(word, 0) for word in words if word.isalpha())
            sentence_scores[sentence] = score
    return sentence_scores


def summarize(text: str, num_sentences: int = 3) -> str:
    """
    Generate an extractive summary of the given text.

    Steps:
    1. Tokenize the text into sentences.
    2. Clean the text and compute word frequencies.
    3. Score each sentence.
    4. Pick the top N sentences (preserving original order).

    Parameters
    ----------
    text : str
        The article or passage to summarize.
    num_sentences : int
        Number of sentences to include in the summary (default: 3).

    Returns
    -------
    str
        The generated summary.
    """
    # Step 1 — Split into sentences using NLTK's sentence tokenizer
    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        # Text is already short enough; return as-is
        return text

    # Step 2 — Compute word frequencies from cleaned text
    cleaned = clean_text(text)
    word_freq = compute_word_frequencies(cleaned)

    # Step 3 — Score each sentence
    sentence_scores = score_sentences(sentences, word_freq)

    # Step 4 — Pick the top N sentences by score
    top_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)

    # Preserve the original reading order
    summary_sentences = [s for s in sentences if s in top_sentences]
    return " ".join(summary_sentences)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

SAMPLE_ARTICLE = """
Artificial intelligence (AI) is intelligence demonstrated by machines,
as opposed to the natural intelligence displayed by animals and humans.
AI research has been defined as the field of study of intelligent agents,
which refers to any system that perceives its environment and takes actions
that maximize its chance of achieving its goals.

The term "artificial intelligence" had previously been used to describe
machines that mimic and display human cognitive skills associated with the
human mind, such as learning and problem solving. This definition has since
been rejected by major AI researchers who now describe AI in terms of
rationality and acting rationally, which does not limit how intelligence
can be articulated.

AI applications include advanced web search engines, recommendation systems
(such as those used by YouTube, Amazon, and Netflix), understanding human speech
(such as Siri and Alexa), self-driving cars, generative AI tools (such as
ChatGPT and AI art), automated decision-making, and competing at the highest
level in strategic game systems such as chess and Go.

As machines become increasingly capable, tasks considered to require intelligence
are often removed from the definition of AI, a phenomenon known as the AI effect.
For instance, optical character recognition is frequently excluded from things
considered to be AI, having become a routine technology.

Artificial intelligence was founded as an academic discipline in 1956, and in the
years since it has experienced several waves of optimism, followed by disappointment
and the loss of funding, followed by new approaches, success, and renewed funding.
AI research has tried and discarded many different approaches, including simulating
the brain, modelling human problem solving, formal logic, large databases of
knowledge, and imitating animal behaviour.
"""


def main():
    """Run a demonstration of the text summarizer."""
    print("=" * 65)
    print("       TEXT SUMMARIZER — Extractive NLP Demo")
    print("=" * 65)

    print(f"\nOriginal article: {len(SAMPLE_ARTICLE.split())} words, "
          f"{len(sent_tokenize(SAMPLE_ARTICLE))} sentences\n")

    for n in [2, 3, 4]:
        summary = summarize(SAMPLE_ARTICLE, num_sentences=n)
        print(f"--- {n}-sentence summary ---")
        print(summary)
        print()

    # Show word frequencies for the top 10 words
    cleaned = clean_text(SAMPLE_ARTICLE)
    word_freq = compute_word_frequencies(cleaned)
    top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]

    print("Top 10 most important words (by normalized frequency):")
    print(f"{'Word':<20} {'Score':>6}")
    print("-" * 28)
    for word, score in top_words:
        print(f"{word:<20} {score:>6.2f}")


if __name__ == "__main__":
    main()
