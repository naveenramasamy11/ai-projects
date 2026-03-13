# 🌸 Project 04 — Iris Flower Classifier

> Classify iris flowers into 3 species using K-Nearest Neighbors and Decision Tree algorithms.

---

## File Table

| File | Purpose |
|------|---------|
| `iris_flower_classifier.py` | Self-contained Python script — run to train, evaluate, and save figures |
| `iris_flower_classifier.ipynb` | Step-by-step Jupyter notebook with explanations and experiments |
| `requirements.txt` | Pinned pip dependencies |
| `iris_results.png` | Output figure (confusion matrices + decision boundary + tree diagram) |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the script
python iris_flower_classifier.py
```

The script prints accuracy/classification reports and saves `iris_results.png`.

To explore interactively:
```bash
jupyter notebook iris_flower_classifier.ipynb
```

---

## Example Output

```
🌸 Iris Flower Classifier
   Comparing KNN vs Decision Tree

Dataset shape : (150, 4)
Classes       : ['setosa', 'versicolor', 'virginica']

==================================================
  K-Nearest Neighbors (k=5)
==================================================
  Accuracy : 97.37%

              precision    recall  f1-score   support
     setosa       1.00      1.00      1.00        13
 versicolor       0.93      1.00      0.96        13
  virginica       1.00      0.92      0.96        12

==================================================
  Decision Tree (max_depth=4)
==================================================
  Accuracy : 97.37%

🌼 New sample [5.1, 3.5, 1.4, 0.2]
   KNN predicts           : setosa
   Decision Tree predicts : setosa

📊 Figure saved to: iris_results.png
✅ Done!
```

---

## Key Concepts

| Concept | Plain English |
|---------|--------------|
| Multi-class classification | Predict which of 3+ categories a sample belongs to |
| Train/test split | Hold out some data to evaluate the model on unseen examples |
| Stratified split | Keep the same class proportions in both train and test |
| Feature scaling (StandardScaler) | Rescale each feature to mean=0, std=1 so distances are fair for KNN |
| K-Nearest Neighbors (KNN) | Classify by majority vote of the *k* closest training samples |
| Decision Tree | Learn a hierarchy of if/else rules from training data |
| `max_depth` | Cap the number of levels in the tree to avoid overfitting |
| Confusion matrix | Grid showing how many samples of each true class were predicted as each class |
| Precision / Recall / F1 | Metrics that go beyond simple accuracy for imbalanced classes |

---

## Libraries Used

| Library | Link |
|---------|------|
| scikit-learn | https://scikit-learn.org |
| matplotlib | https://matplotlib.org |
| seaborn | https://seaborn.pydata.org |
| numpy | https://numpy.org |
| pandas | https://pandas.pydata.org |

---

*Part of the **Daily AI Projects** series — one beginner ML project per weekday.*
