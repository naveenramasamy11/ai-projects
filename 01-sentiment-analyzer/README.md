# 🎭 Project 01: Sentiment Analyzer

Analyze whether any piece of text is **Positive**, **Negative**, or **Neutral** — with zero training required!

## 🧠 What It Does

Uses two pre-trained NLP tools:
- **VADER** — tuned for social media, short sentences, emojis, slang
- **TextBlob** — great for general text; also provides a *subjectivity* score

## 📁 Files

| File | Description |
|------|-------------|
| `sentiment_analyzer.py` | Ready-to-run Python script |
| `sentiment_analyzer.ipynb` | Step-by-step Jupyter Notebook (recommended for learning) |
| `requirements.txt` | Python dependencies |

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the script
python sentiment_analyzer.py
```

Or open the notebook:
```bash
jupyter notebook sentiment_analyzer.ipynb
```

## 📊 Example Output

```
Text: "I absolutely LOVE this product! It's amazing!"
  [VADER]    Label: Positive    Compound Score: +0.836
  [TextBlob] Label: Positive    Polarity: +0.600  Subjectivity: 0.900

Text: "This is the worst experience I've ever had."
  [VADER]    Label: Negative    Compound Score: -0.679
  [TextBlob] Label: Negative    Polarity: -1.000  Subjectivity: 1.000

Text: "The package arrived on time."
  [VADER]    Label: Neutral     Compound Score: +0.000
  [TextBlob] Label: Neutral     Polarity: +0.000  Subjectivity: 0.000
```

## 🔑 Key Concepts

- **Compound score (VADER):** -1.0 = very negative, 0 = neutral, +1.0 = very positive
- **Polarity (TextBlob):** Same scale as VADER compound
- **Subjectivity (TextBlob):** 0.0 = factual/objective, 1.0 = very opinionated

## 🛠 Libraries Used

- [`nltk`](https://www.nltk.org/) — VADER sentiment model
- [`textblob`](https://textblob.readthedocs.io/) — TextBlob sentiment model
- [`pandas`](https://pandas.pydata.org/) — Data tables
- [`matplotlib`](https://matplotlib.org/) — Charts
