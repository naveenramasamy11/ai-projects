"""
Movie Recommendation System
============================
A beginner-friendly recommendation engine that suggests similar movies
based on their genre tags using TF-IDF vectorization and cosine similarity.

Concepts covered:
  - Content-based filtering
  - TF-IDF (Term Frequency–Inverse Document Frequency)
  - Cosine similarity
  - Pandas DataFrames

Usage:
  python movie_recommendation.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Dataset — 30 well-known movies with genre/keyword tags
# ---------------------------------------------------------------------------

MOVIES = [
    {"title": "The Dark Knight",       "tags": "action crime thriller superhero batman villain"},
    {"title": "Inception",             "tags": "sci-fi thriller dream heist mind-bending"},
    {"title": "Interstellar",          "tags": "sci-fi space adventure drama time-travel"},
    {"title": "The Matrix",            "tags": "sci-fi action thriller virtual-reality dystopia"},
    {"title": "Avengers: Endgame",     "tags": "action superhero marvel adventure"},
    {"title": "The Godfather",         "tags": "crime drama mafia family classic"},
    {"title": "Pulp Fiction",          "tags": "crime thriller drama cult non-linear"},
    {"title": "Forrest Gump",          "tags": "drama romance comedy history inspirational"},
    {"title": "The Lion King",         "tags": "animation family drama adventure musical"},
    {"title": "Toy Story",             "tags": "animation family comedy adventure friendship"},
    {"title": "Finding Nemo",          "tags": "animation family adventure comedy ocean"},
    {"title": "Up",                    "tags": "animation family adventure drama friendship"},
    {"title": "Shrek",                 "tags": "animation comedy family fairy-tale adventure"},
    {"title": "The Avengers",          "tags": "action superhero marvel adventure team"},
    {"title": "Iron Man",              "tags": "action superhero marvel tech billionaire"},
    {"title": "Spider-Man: No Way Home","tags": "action superhero marvel spider-man multiverse"},
    {"title": "Doctor Strange",        "tags": "action superhero marvel magic multiverse"},
    {"title": "Guardians of the Galaxy","tags": "action sci-fi marvel comedy adventure space"},
    {"title": "Gravity",               "tags": "sci-fi space thriller survival drama"},
    {"title": "The Martian",           "tags": "sci-fi space adventure survival drama comedy"},
    {"title": "Mad Max: Fury Road",    "tags": "action sci-fi thriller post-apocalyptic survival"},
    {"title": "John Wick",             "tags": "action thriller crime assassin revenge"},
    {"title": "Taken",                 "tags": "action thriller crime rescue revenge"},
    {"title": "The Silence of the Lambs","tags": "thriller crime horror psychological detective"},
    {"title": "Se7en",                 "tags": "thriller crime mystery detective dark"},
    {"title": "Fight Club",            "tags": "drama thriller cult psychology identity"},
    {"title": "The Shawshank Redemption","tags": "drama crime hope friendship classic"},
    {"title": "Schindler's List",      "tags": "drama history war hope tragedy"},
    {"title": "Titanic",               "tags": "romance drama history tragedy disaster"},
    {"title": "La La Land",            "tags": "romance drama musical comedy ambition"},
]


def build_dataframe(movies: list[dict]) -> pd.DataFrame:
    """Convert the movies list into a Pandas DataFrame.

    Parameters
    ----------
    movies : list of dict
        Each dict must have 'title' and 'tags' keys.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by integer with 'title' and 'tags' columns.
    """
    return pd.DataFrame(movies)


def build_similarity_matrix(df: pd.DataFrame) -> np.ndarray:
    """Compute a cosine-similarity matrix from movie tags using TF-IDF.

    Steps
    -----
    1. TfidfVectorizer converts each movie's tag string into a numeric vector.
       Words that appear in many movies get lower weight (IDF part).
    2. cosine_similarity measures how similar two vectors are.
       Score of 1.0 = identical direction, 0.0 = completely different.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'tags' column.

    Returns
    -------
    np.ndarray
        Square similarity matrix of shape (n_movies, n_movies).
    """
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["tags"])
    similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return similarity


def get_recommendations(title: str, df: pd.DataFrame,
                        similarity: np.ndarray, top_n: int = 5) -> pd.DataFrame:
    """Return the top-N most similar movies for a given title.

    Parameters
    ----------
    title : str
        The movie title to base recommendations on.
    df : pd.DataFrame
        The full movies DataFrame.
    similarity : np.ndarray
        Pre-computed cosine-similarity matrix.
    top_n : int
        How many recommendations to return (default 5).

    Returns
    -------
    pd.DataFrame
        DataFrame with 'title', 'tags', and 'similarity_score' columns,
        sorted descending by similarity score.

    Raises
    ------
    ValueError
        If the title is not found in the dataset.
    """
    # Find the integer index of the movie
    matches = df[df["title"].str.lower() == title.lower()]
    if matches.empty:
        raise ValueError(f"Movie '{title}' not found. Check the title and try again.")

    movie_idx = matches.index[0]

    # Get similarity scores for every other movie
    scores = list(enumerate(similarity[movie_idx]))

    # Sort by score descending; skip index 0 (the movie itself)
    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
    top_scores = scores_sorted[1: top_n + 1]   # exclude the input movie

    # Build result DataFrame
    indices = [i for i, _ in top_scores]
    similarity_values = [round(score, 4) for _, score in top_scores]

    result = df.iloc[indices][["title", "tags"]].copy()
    result["similarity_score"] = similarity_values
    result.reset_index(drop=True, inplace=True)
    return result


def plot_recommendations(recommendations: pd.DataFrame, query_title: str,
                         save_path: str = "recommendations.png") -> None:
    """Create a horizontal bar chart of recommendation scores and save it.

    Parameters
    ----------
    recommendations : pd.DataFrame
        Output from get_recommendations().
    query_title : str
        The movie the recommendations are based on (used in title).
    save_path : str
        File path where the PNG will be saved.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    # Reverse order so highest score appears at top
    titles = recommendations["title"].tolist()[::-1]
    scores = recommendations["similarity_score"].tolist()[::-1]

    colors = plt.cm.Blues(np.linspace(0.4, 0.85, len(titles)))

    bars = ax.barh(titles, scores, color=colors, edgecolor="steelblue", linewidth=0.6)

    # Annotate each bar with the score
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}", va="center", fontsize=9)

    ax.set_xlabel("Cosine Similarity Score", fontsize=11)
    ax.set_title(f'Top Recommendations for  "{query_title}"', fontsize=13, fontweight="bold")
    ax.set_xlim(0, max(scores) * 1.25)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Chart saved → {save_path}")


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main() -> None:
    """Run a demonstration of the movie recommendation system."""
    print("=" * 60)
    print("        Movie Recommendation System (Content-Based)")
    print("=" * 60)

    # Build data
    df = build_dataframe(MOVIES)
    print(f"\nLoaded {len(df)} movies.")

    # Build similarity matrix
    similarity = build_similarity_matrix(df)
    print("Cosine-similarity matrix built.\n")

    # Demo queries
    demo_titles = ["Inception", "The Lion King", "John Wick"]

    for query in demo_titles:
        print(f"{'─' * 50}")
        print(f" Query movie : {query}")
        print(f"{'─' * 50}")

        recs = get_recommendations(query, df, similarity, top_n=5)
        print(recs[["title", "similarity_score"]].to_string(index=False))
        print()

    # Save chart for the first demo
    recs_chart = get_recommendations(demo_titles[0], df, similarity, top_n=5)
    plot_recommendations(recs_chart, demo_titles[0], save_path="recommendations.png")

    print("\nDone! Explore the similarity matrix or try your own movie titles.")


if __name__ == "__main__":
    main()
