# 🔢 Project 08 — Digit Recognizer

> Train a neural network to recognise handwritten digits (0–9) using the scikit-learn MNIST-like digits dataset.

---

## 📁 Files

| File | Purpose |
|------|---------|
| `digit_recognizer.py` | Main Python script — full ML pipeline from data to saved model |
| `digit_recognizer.ipynb` | Step-by-step Jupyter notebook with explanations and exercises |
| `requirements.txt` | Python package dependencies |
| `sample_digits.png` | *(generated)* Grid of sample digit images from the dataset |
| `digit_confusion_matrix.png` | *(generated)* Heatmap showing correct vs incorrect predictions |
| `digit_predictions.png` | *(generated)* Side-by-side view of predicted vs true labels |
| `digit_recognizer_model.pkl` | *(generated)* Saved trained MLP model |
| `digit_scaler.pkl` | *(generated)* Saved StandardScaler |

---

## ⚡ Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the script
python digit_recognizer.py
```

Or open `digit_recognizer.ipynb` in JupyterLab / VS Code and run all cells.

---

## 📊 Example Output

```
Dataset loaded: 1797 samples, 64 features (8×8 pixels), 10 classes (0–9)
Train samples: 1437 | Test samples: 360
Training the neural network (MLP)…
Training complete.

Test Accuracy: 97.78%

Classification Report:
              precision    recall  f1-score   support
           0       1.00      1.00      1.00        36
           1       0.97      0.97      0.97        36
           2       0.97      1.00      0.99        36
           3       1.00      0.97      0.99        36
           4       0.97      1.00      0.99        36
           5       0.97      1.00      0.99        36
           6       1.00      1.00      1.00        36
           7       1.00      0.97      0.99        36
           8       0.94      0.94      0.94        36
           9       1.00      0.94      0.97        36

    accuracy                           0.98       360
```

---

## 🧠 Key Concepts

| Concept | Plain English |
|---------|---------------|
| **Pixel array** | An image is a grid of numbers — each number is a brightness value |
| **Flattening** | Convert a 2-D image (8×8) into a 1-D feature vector (64 values) |
| **StandardScaler** | Rescales features to mean=0, std=1 so the network trains smoothly |
| **Data leakage** | Fitting the scaler on test data leaks info — always fit on train only |
| **MLP (Neural Network)** | Layers of connected neurons that learn patterns via backpropagation |
| **ReLU** | Activation function: output = max(0, x) — adds non-linearity |
| **Adam optimiser** | Adaptive gradient descent — adjusts learning rate per parameter |
| **Confusion matrix** | Table showing which digit classes get mixed up |
| **joblib** | Python library for saving/loading trained ML models to disk |

---

## 📦 Libraries Used

| Library | Purpose | Link |
|---------|---------|------|
| scikit-learn | Dataset, MLP, metrics | [scikit-learn.org](https://scikit-learn.org) |
| matplotlib | Plotting charts | [matplotlib.org](https://matplotlib.org) |
| seaborn | Confusion matrix heatmap | [seaborn.pydata.org](https://seaborn.pydata.org) |
| numpy | Numerical arrays | [numpy.org](https://numpy.org) |
| joblib | Model persistence | [joblib.readthedocs.io](https://joblib.readthedocs.io) |

---

⭐ **Difficulty:** Beginner–Intermediate
🕒 **Estimated time:** 30–45 minutes
