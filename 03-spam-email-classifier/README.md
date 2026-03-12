# 🛡️ Project 03 — Spam Email Classifier

> Classify email messages as **spam** or **ham** (legitimate) using Naive Bayes and TF-IDF.

---

## 📁 Files

| File | Purpose |
|---|---|
| `spam_email_classifier.py` | Standalone Python script — run this to see the full pipeline |
| `spam_email_classifier.ipynb` | Step-by-step Jupyter notebook with explanations |
| `requirements.txt` | All pip-installable dependencies |
| `confusion_matrix.png` | Generated chart — saved when you run the script or notebook |

---

## ⚡ Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the script
python spam_email_classifier.py

# 3. Or open the notebook
jupyter notebook spam_email_classifier.ipynb
```

---

## 🖥️ Example Output

```
============================================================
   Spam Email Classifier — Naive Bayes Demo
============================================================

[1/5] Loading dataset …
  Total messages : 50
  Ham (legit)    : 26
  Spam           : 24

[3/5] Extracting TF-IDF features …
  Vocabulary size  : 357 terms

[5/5] Evaluating on test set …

  Accuracy : 90.0%

  Top 10 words most indicative of SPAM:
    1. 'free'                          (log-prob diff: +2.85)
    2. 'click'                         (log-prob diff: +2.61)
    3. 'guaranteed'                    (log-prob diff: +2.44)
    ...

  Live Predictions:
  Message                                                 Result    Spam%
  -------------------------------------------------------------------------
  URGENT: Your bank account has been locked. Verify n…   SPAM 🚨   97.3%
  Hey, just checking if you got my last email about t…   Ham  ✅    2.1%
```

---

## 🧠 Key Concepts

| Concept | Plain English |
|---|---|
| TF-IDF | Converts words to numbers; rare-but-important words score higher |
| Bigrams | Two-word phrases like "click here" captured as single features |
| Multinomial Naive Bayes | Fast probabilistic classifier ideal for word-frequency data |
| Laplace Smoothing | Prevents zero-probability crashes for unseen words |
| Confusion Matrix | Visual grid of correct vs. incorrect predictions |
| Precision | "Of flagged spam, how much was actually spam?" |
| Recall | "Of real spam, how much did we catch?" |

---

## 📦 Libraries Used

| Library | Link | Purpose |
|---|---|---|
| scikit-learn | https://scikit-learn.org | TF-IDF, Naive Bayes, metrics |
| pandas | https://pandas.pydata.org | DataFrame handling |
| numpy | https://numpy.org | Array operations |
| matplotlib | https://matplotlib.org | Plotting |
| seaborn | https://seaborn.pydata.org | Styled confusion matrix heatmap |

---

*Part of the [AI Projects](https://github.com/naveenramasamy11/ai-projects) beginner series.*
