"""
=============================================================
  Project 01: Sentiment Analyzer
  ---------------------------------
  This script analyzes the sentiment (positive / negative /
  neutral) of any text you give it, using two approaches:
    1. VADER  — great for social media & short text
    2. TextBlob — great for general text

  No machine-learning training required — both models are
  pre-trained and ready to use out of the box!
=============================================================
"""

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt

# ── Download required NLTK data (only needed once) ──────────
nltk.download("vader_lexicon", quiet=True)


# ── Helper functions ─────────────────────────────────────────

def analyze_with_vader(text: str) -> dict:
    """
    Analyze sentiment using VADER (Valence Aware Dictionary
    and sEntiment Reasoner).

    Returns a dict with:
      - compound : overall score from -1 (most negative)
                   to +1 (most positive)
      - label    : 'Positive', 'Negative', or 'Neutral'
    """
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return {"compound": compound, "label": label, "raw_scores": scores}


def analyze_with_textblob(text: str) -> dict:
    """
    Analyze sentiment using TextBlob.

    Returns a dict with:
      - polarity    : -1.0 (negative) to 1.0 (positive)
      - subjectivity: 0.0 (objective) to 1.0 (subjective)
      - label       : 'Positive', 'Negative', or 'Neutral'
    """
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    if polarity > 0:
        label = "Positive"
    elif polarity < 0:
        label = "Negative"
    else:
        label = "Neutral"

    return {"polarity": polarity, "subjectivity": subjectivity, "label": label}


def analyze_text(text: str) -> None:
    """Print a full sentiment report for the given text."""
    print("\n" + "=" * 60)
    print(f"  Text: {text[:80]}{'...' if len(text) > 80 else ''}")
    print("=" * 60)

    vader_result = analyze_with_vader(text)
    tb_result = analyze_with_textblob(text)

    print(f"\n  [VADER]    Label: {vader_result['label']:10}  "
          f"Compound Score: {vader_result['compound']:+.3f}")
    print(f"  [TextBlob] Label: {tb_result['label']:10}  "
          f"Polarity: {tb_result['polarity']:+.3f}  "
          f"Subjectivity: {tb_result['subjectivity']:.3f}")
    print()


def batch_analyze(texts: list[str]) -> pd.DataFrame:
    """
    Analyze a list of texts and return a summary DataFrame.
    Also displays a bar chart of sentiment distribution.
    """
    records = []
    for text in texts:
        vader = analyze_with_vader(text)
        tb = analyze_with_textblob(text)
        records.append({
            "text": text[:60] + ("..." if len(text) > 60 else ""),
            "vader_label": vader["label"],
            "vader_score": vader["compound"],
            "textblob_label": tb["label"],
            "textblob_polarity": tb["polarity"],
        })

    df = pd.DataFrame(records)

    # ── Plot sentiment distribution ──────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Sentiment Distribution", fontsize=14, fontweight="bold")

    for ax, col, title in zip(
        axes,
        ["vader_label", "textblob_label"],
        ["VADER", "TextBlob"],
    ):
        counts = df[col].value_counts()
        colors = {"Positive": "#4CAF50", "Neutral": "#2196F3", "Negative": "#F44336"}
        ax.bar(counts.index, counts.values,
               color=[colors.get(l, "gray") for l in counts.index])
        ax.set_title(f"{title} Sentiment")
        ax.set_ylabel("Count")
        ax.set_xlabel("Sentiment")

    plt.tight_layout()
    plt.savefig("sentiment_distribution.png", dpi=120)
    print("\nChart saved to sentiment_distribution.png")
    plt.show()

    return df


# ── Main demo ────────────────────────────────────────────────

if __name__ == "__main__":
    # --- Single text examples ---
    sample_texts = [
        "I absolutely love this product! It's amazing and works perfectly.",
        "This is the worst experience I've ever had. Totally disappointed.",
        "The package arrived on time. It was okay.",
        "I'm so happy with the results! Couldn't be better!",
        "The movie was boring and way too long.",
        "Not sure how I feel about this. It's just... fine.",
    ]

    print("\n=== SINGLE TEXT ANALYSIS ===")
    for text in sample_texts[:3]:
        analyze_text(text)

    print("\n=== BATCH ANALYSIS ===")
    df = batch_analyze(sample_texts)
    print("\nSummary Table:")
    print(df.to_string(index=False))

    # --- Interactive mode ---
    print("\n" + "=" * 60)
    print("  TRY IT YOURSELF!")
    print("  Type any sentence and press Enter to analyze it.")
    print("  Type 'quit' to exit.")
    print("=" * 60)

    while True:
        user_input = input("\nEnter text: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if user_input:
            analyze_text(user_input)
