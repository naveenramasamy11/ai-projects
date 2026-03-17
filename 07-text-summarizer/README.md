# Project 07 – Text Summarizer

> Extract the most important sentences from any article using word frequency scoring.

---

## Files

| File | Purpose |
|------|---------|
| `text_summarizer.py` | Main Python script — run for a live demo |
| `text_summarizer.ipynb` | Step-by-step Jupyter notebook with explanations |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the demo
python text_summarizer.py
```

---

## Example Output

```
=================================================================
       TEXT SUMMARIZER — Extractive NLP Demo
=================================================================

Original article: 187 words, 13 sentences

--- 3-sentence summary ---
AI research has been defined as the field of study of intelligent agents,
which refers to any system that perceives its environment and takes actions
that maximize its chance of achieving its goals. AI applications include
advanced web search engines, recommendation systems... Artificial intelligence
was founded as an academic discipline in 1956...

Top 10 most important words (by normalized frequency):
Word                  Score
----------------------------
intelligence           1.00
ai                     0.88
research               0.50
human                  0.50
machines               0.38
...
```

---

## Key Concepts

| Concept | Plain English |
|---------|--------------|
| Tokenization | Splitting text into words or sentences |
| Stop words | Common filler words (the, is, in) removed before analysis |
| Word frequency | Counting how often each word appears |
| Normalization | Scaling counts to a 0–1 range for fair comparison |
| Sentence scoring | Each sentence is scored by summing its word frequencies |
| Extractive summarization | Picking existing sentences — no new text is generated |
| Abstractive summarization | Generating brand-new sentences (needs LLMs like GPT) |

---

## Libraries Used

| Library | Purpose | Link |
|---------|---------|------|
| [NLTK](https://www.nltk.org/) | Tokenization, stop words | https://www.nltk.org |
| [heapq](https://docs.python.org/3/library/heapq.html) | Efficiently find top-N sentences | Python stdlib |
| [collections.defaultdict](https://docs.python.org/3/library/collections.html) | Word frequency counting | Python stdlib |

---

## Difficulty

⭐ Beginner — no ML model required, pure NLP with frequency statistics.
