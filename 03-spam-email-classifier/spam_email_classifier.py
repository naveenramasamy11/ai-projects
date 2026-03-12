"""
Spam Email Classifier
=====================
A beginner-friendly demonstration of text classification using the Naive Bayes
algorithm. This script trains a Multinomial Naive Bayes model to distinguish
between spam and legitimate (ham) email messages.

Key Concepts:
    - Text preprocessing (lowercasing, stopword removal)
    - Bag-of-Words representation with CountVectorizer / TF-IDF
    - Multinomial Naive Bayes classifier
    - Evaluation: accuracy, precision, recall, confusion matrix

Usage:
    python spam_email_classifier.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ---------------------------------------------------------------------------
# 1.  Dataset
# ---------------------------------------------------------------------------

# Inline dataset — no external downloads needed.
# Each tuple is (label, message): "spam" or "ham".
RAW_DATA = [
    # --- ham (legitimate) ---
    ("ham", "Hey, are you free for lunch tomorrow?"),
    ("ham", "Please review the attached quarterly report before the meeting."),
    ("ham", "Your package has been shipped and will arrive on Friday."),
    ("ham", "Can you send me the project files when you get a chance?"),
    ("ham", "Meeting rescheduled to 3 PM — same room."),
    ("ham", "Happy birthday! Hope you have a wonderful day."),
    ("ham", "I will be out of office until Monday. For urgent matters contact Sarah."),
    ("ham", "Thanks for the update. I will take a look tonight."),
    ("ham", "The code review comments have been addressed. PR is ready for merge."),
    ("ham", "Reminder: team standup at 10 AM today."),
    ("ham", "Could you double-check the numbers in the spreadsheet?"),
    ("ham", "The flight is confirmed. Boarding at gate 12."),
    ("ham", "Great work on the presentation! The client loved it."),
    ("ham", "Please submit your timesheets by end of day Friday."),
    ("ham", "I found a great recipe — shall I send it over?"),
    ("ham", "Your doctor appointment is confirmed for Tuesday at 2 PM."),
    ("ham", "Library book renewal successful. Due date extended by 3 weeks."),
    ("ham", "Call me when you land. Safe travels!"),
    ("ham", "The patch has been deployed to production successfully."),
    ("ham", "Budget approval received. You can proceed with the purchase."),
    # --- spam ---
    ("spam", "CONGRATULATIONS! You have won a $1000 Walmart gift card. Click here NOW!"),
    ("spam", "Earn $500 per day working from home. No experience required. Sign up free!"),
    ("spam", "URGENT: Your account has been compromised. Verify your password immediately."),
    ("spam", "Limited time offer! Buy 1 get 3 FREE. Order before midnight!"),
    ("spam", "You have been selected for a special loan offer. Apply now, no credit check!"),
    ("spam", "HOT SINGLES in your area want to meet you. Click to view profiles!"),
    ("spam", "FREE iPhone 15 Pro! You are our lucky winner. Claim your prize today."),
    ("spam", "Make money fast! Proven system earns $10,000 monthly. Join thousands!"),
    ("spam", "Your PayPal account is suspended. Log in immediately to restore access."),
    ("spam", "Lose 20 lbs in 2 weeks! Miracle pill doctors don't want you to know about."),
    ("spam", "Cheap medication online — no prescription needed. Huge discounts!"),
    ("spam", "Double your investment in 24 hours. 100% guaranteed returns. Act now!"),
    ("spam", "You owe a tax refund. Click the link to claim $3,200 from the IRS."),
    ("spam", "Enlarge and enhance — guaranteed results or your money back!"),
    ("spam", "Final notice: your subscription expires today. Renew for FREE access!"),
    ("spam", "Click HERE to unsubscribe from our mailing list. Or win a cruise!"),
    ("spam", "Exclusive VIP offer: Casino bonus $500 FREE. Register in 60 seconds!"),
    ("spam", "Your Microsoft account requires immediate verification. Click to confirm."),
    ("spam", "Stock tip: buy XYZ shares now before the price explodes tomorrow!"),
    ("spam", "Congratulations winner! Collect your Amazon voucher worth $250 now!"),
    # Additional samples to improve model balance
    ("ham", "See you at the conference next week."),
    ("ham", "The invoice for March has been processed."),
    ("ham", "Jenkins build #472 passed all tests."),
    ("ham", "Dinner reservation confirmed for Saturday at 7 PM."),
    ("spam", "Act fast — only 3 spots left in our wealth secrets webinar!"),
    ("spam", "You have a pending wire transfer of $8,500. Confirm your details."),
    ("spam", "Surprise! Click to reveal your exclusive reward from our survey."),
    ("spam", "SALE 90% off designer brands. Today only. Shop now!"),
    ("ham", "The kids recital starts at 6 PM. Bring the camera!"),
    ("ham", "Annual performance reviews begin next week. Schedule your 1-on-1s."),
]


def build_dataframe(raw_data: list[tuple[str, str]]) -> pd.DataFrame:
    """Convert raw (label, message) tuples into a tidy DataFrame.

    Args:
        raw_data: List of (label, message) tuples.

    Returns:
        DataFrame with columns ['label', 'message', 'is_spam'].
    """
    df = pd.DataFrame(raw_data, columns=["label", "message"])
    df["is_spam"] = (df["label"] == "spam").astype(int)
    return df


# ---------------------------------------------------------------------------
# 2.  Feature Extraction
# ---------------------------------------------------------------------------

def build_vectorizer() -> TfidfVectorizer:
    """Create a TF-IDF vectorizer with sensible defaults for short texts.

    TF-IDF (Term Frequency–Inverse Document Frequency) gives higher weight
    to words that appear often in one document but rarely across all documents,
    helping the model focus on discriminative vocabulary.

    Returns:
        Configured TfidfVectorizer instance.
    """
    return TfidfVectorizer(
        lowercase=True,        # normalise case
        stop_words="english",  # remove common words like "the", "is", "at"
        max_features=500,      # keep only the top-500 most informative terms
        ngram_range=(1, 2),    # use single words AND two-word phrases
    )


# ---------------------------------------------------------------------------
# 3.  Model
# ---------------------------------------------------------------------------

def train_model(X_train, y_train):
    """Train a Multinomial Naive Bayes classifier.

    Naive Bayes is fast, interpretable, and works surprisingly well for
    text classification tasks — especially with limited data.

    Args:
        X_train: Sparse TF-IDF feature matrix for training samples.
        y_train: Binary labels (0 = ham, 1 = spam).

    Returns:
        Fitted MultinomialNB model.
    """
    model = MultinomialNB(alpha=1.0)   # alpha=1.0 is Laplace smoothing
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# 4.  Evaluation & Visualisation
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, output_path: str) -> None:
    """Save a colour-coded confusion matrix to a PNG file.

    Args:
        y_true: Actual labels.
        y_pred: Predicted labels.
        output_path: File path for the saved chart.
    """
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Ham (Legit)", "Spam"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title("Confusion Matrix — Spam Email Classifier", fontsize=13, pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"  → Confusion matrix saved to: {output_path}")


def show_top_spam_words(model, vectorizer: TfidfVectorizer, n: int = 10) -> None:
    """Print the words most associated with spam by the model.

    Args:
        model: Trained MultinomialNB model.
        vectorizer: Fitted TfidfVectorizer (provides feature names).
        n: Number of top features to display.
    """
    feature_names = vectorizer.get_feature_names_out()
    # log-probability difference: spam class minus ham class
    log_prob_diff = model.feature_log_prob_[1] - model.feature_log_prob_[0]
    top_indices = np.argsort(log_prob_diff)[-n:][::-1]

    print(f"\n  Top {n} words most indicative of SPAM:")
    for rank, idx in enumerate(top_indices, 1):
        print(f"    {rank:2d}. {feature_names[idx]!r:30s}  (log-prob diff: {log_prob_diff[idx]:+.2f})")


# ---------------------------------------------------------------------------
# 5.  Prediction Helper
# ---------------------------------------------------------------------------

def predict_messages(messages: list[str], model, vectorizer: TfidfVectorizer) -> None:
    """Classify a list of new messages and print the results.

    Args:
        messages: Raw email/message strings to classify.
        model: Trained classifier.
        vectorizer: Fitted TfidfVectorizer.
    """
    X_new = vectorizer.transform(messages)
    predictions = model.predict(X_new)
    probas = model.predict_proba(X_new)

    print("\n  Live Predictions:")
    print(f"  {'Message':<55} {'Result':<8}  Spam%")
    print("  " + "-" * 75)
    for msg, pred, proba in zip(messages, predictions, probas):
        label = "SPAM 🚨" if pred == 1 else "Ham  ✅"
        spam_pct = proba[1] * 100
        print(f"  {msg[:54]:<55} {label:<8}  {spam_pct:5.1f}%")


# ---------------------------------------------------------------------------
# 6.  Main
# ---------------------------------------------------------------------------

def main() -> None:
    """End-to-end pipeline: load data → train → evaluate → predict."""

    print("=" * 60)
    print("   Spam Email Classifier — Naive Bayes Demo")
    print("=" * 60)

    # --- Load data ---
    print("\n[1/5] Loading dataset …")
    df = build_dataframe(RAW_DATA)
    spam_count = df["is_spam"].sum()
    ham_count = len(df) - spam_count
    print(f"  Total messages : {len(df)}")
    print(f"  Ham (legit)    : {ham_count}")
    print(f"  Spam           : {spam_count}")

    # --- Split ---
    print("\n[2/5] Splitting into train / test sets (80 / 20) …")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df["message"], df["is_spam"], test_size=0.2, random_state=42, stratify=df["is_spam"]
    )
    print(f"  Training samples : {len(X_train_raw)}")
    print(f"  Test samples     : {len(X_test_raw)}")

    # --- Vectorise ---
    print("\n[3/5] Extracting TF-IDF features …")
    vectorizer = build_vectorizer()
    X_train = vectorizer.fit_transform(X_train_raw)   # learn vocabulary on train
    X_test = vectorizer.transform(X_test_raw)          # apply same vocab to test
    print(f"  Vocabulary size  : {X_train.shape[1]} terms")

    # --- Train ---
    print("\n[4/5] Training Multinomial Naive Bayes …")
    model = train_model(X_train, y_train)
    print("  Model trained successfully.")

    # --- Evaluate ---
    print("\n[5/5] Evaluating on test set …")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Accuracy : {acc * 100:.1f}%\n")
    print("  Classification Report:")
    report = classification_report(y_test, y_pred, target_names=["Ham", "Spam"])
    for line in report.splitlines():
        print(f"    {line}")

    # Confusion matrix chart
    chart_path = os.path.join(os.path.dirname(__file__), "confusion_matrix.png")
    plot_confusion_matrix(y_test, y_pred, chart_path)

    # Interpretability
    show_top_spam_words(model, vectorizer)

    # --- Live demo ---
    demo_messages = [
        "URGENT: Your bank account has been locked. Verify now to restore access!",
        "Hey, just checking if you got my last email about the project deadline.",
        "Congratulations! You won a FREE vacation. Click to claim your prize!",
        "Please find the attached invoice for your review.",
        "Make $10,000 a week from home — guaranteed, no experience needed!",
        "Reminder: performance review scheduled for next Thursday at 10 AM.",
    ]
    predict_messages(demo_messages, model, vectorizer)

    print("\n" + "=" * 60)
    print("   Done! Check confusion_matrix.png for the visualisation.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
