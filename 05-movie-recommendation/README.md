# 🎬 Project 05 — Movie Recommendation System

> Suggest similar movies using TF-IDF vectorization and cosine similarity.

---

## File Table

| File | Purpose |
|------|---------|
| `movie_recommendation.py` | Main Python script — builds recommender & runs demo |
| `movie_recommendation.ipynb` | Step-by-step Jupyter notebook with explanations |
| `requirements.txt` | Pinned Python dependencies |
| `recommendations.png` | Bar-chart output (generated on first run) |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the script
python movie_recommendation.py
```

Or open the notebook for a guided walkthrough:

```bash
jupyter notebook movie_recommendation.ipynb
```

---

## Example Output

```
============================================================
        Movie Recommendation System (Content-Based)
============================================================

Loaded 30 movies.
Cosine-similarity matrix built.

──────────────────────────────────────────────────
 Query movie : Inception
──────────────────────────────────────────────────
             title  similarity_score
         The Matrix            0.4216
     Gravity                   0.3181
  Mad Max: Fury Road           0.2803
      Interstellar             0.2610
          Se7en                0.2310
```

A bar chart `recommendations.png` is also saved automatically.

---

## Key Concepts

| Concept | Plain-English Explanation |
|---------|--------------------------|
| **Content-Based Filtering** | Recommend items based on their own features (tags, genres), not user ratings |
| **TF-IDF** | Converts words to numbers; rare keywords get higher weight than common ones |
| **Cosine Similarity** | Measures the angle between two vectors — score close to 1.0 means very similar |
| **Vectorization** | Turning raw text into numeric arrays so Python can do math on it |
| **Collaborative Filtering** | The next step — uses *user ratings* to recommend (powers Netflix, Spotify) |

---

## Libraries Used

| Library | Purpose | Link |
|---------|---------|------|
| pandas | Data tables | [pandas.pydata.org](https://pandas.pydata.org) |
| scikit-learn | TF-IDF + cosine similarity | [scikit-learn.org](https://scikit-learn.org) |
| matplotlib | Bar chart visualization | [matplotlib.org](https://matplotlib.org) |
| numpy | Numeric arrays | [numpy.org](https://numpy.org) |
