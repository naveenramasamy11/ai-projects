# 🌦️ Project 09 – Weather Classifier

> Predict weather type (Sunny / Rainy / Cloudy / Snowy) from meteorological features using a **Random Forest** classifier.

---

## 📁 Files

| File | Purpose |
|---|---|
| `weather_classifier.py` | Self-contained Python script — runs end-to-end demo |
| `weather_classifier.ipynb` | Step-by-step Jupyter notebook with explanations |
| `requirements.txt` | Pinned dependencies |
| `weather_classifier_results.png` | Chart output (generated on first run) |

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the script
python weather_classifier.py

# Or open the notebook
jupyter notebook weather_classifier.ipynb
```

---

## 📊 Example Output

```
======================================================
  Weather Classifier — Random Forest Demo
======================================================

[1/4] Generating synthetic weather dataset …
      1000 rows | {'Sunny': 512, 'Cloudy': 301, 'Rainy': 149, 'Snowy': 38}

[2/4] Training Random Forest classifier …

[3/4] Evaluation results:
      Accuracy : 99.00%

--- Classification Report ---
              precision    recall  f1-score   support
      Cloudy       0.98      0.98      0.98        60
       Rainy       1.00      1.00      1.00        30
       Snowy       1.00      1.00      1.00         8
       Sunny       0.99      0.99      0.99       102

      5-Fold CV Accuracy: 98.90% ± 0.32%

--- Quick Prediction Demo ---
  Input  : 28°C, 45% humidity, 15 km/h wind, 20% cloud, 1018 hPa
  Prediction : Sunny
  Probabilities : {'Cloudy': '0%', 'Rainy': '0%', 'Snowy': '0%', 'Sunny': '100%'}
```

---

## 🧠 Key Concepts

| Concept | Plain English |
|---|---|
| **Random Forest** | An ensemble of decision trees that vote — more accurate and robust than a single tree |
| **Feature Engineering** | Deriving new, more informative columns (heat index, discomfort score) from raw measurements |
| **Train / Test Split** | Holding out 20% of data to evaluate generalisation |
| **Confusion Matrix** | Grid of actual vs. predicted classes — diagonal = correct predictions |
| **Cross-Validation** | Repeated train/test over 5 folds for a robust accuracy estimate |
| **Feature Importance** | Which columns the model relied on most when making decisions |

---

## 📦 Libraries Used

| Library | Purpose | Link |
|---|---|---|
| scikit-learn | Random Forest, metrics, preprocessing | https://scikit-learn.org |
| pandas | Data wrangling | https://pandas.pydata.org |
| numpy | Numerical arrays | https://numpy.org |
| matplotlib | Charts | https://matplotlib.org |
| seaborn | Heatmaps | https://seaborn.pydata.org |

---

## 💡 Difficulty: ⭐⭐ Beginner–Intermediate
