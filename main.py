import os
import ast
import requests
import pandas as pd

from flask import Flask, request, jsonify, render_template
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Flask app setup ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

# ---------- OMDb API key (set in Render / env) ----------
OMDB_API_KEY = os.environ.get("OMDB_API_KEY", None)

# ---------- Load and prepare data ----------
movies = pd.read_csv(os.path.join(BASE_DIR, "tmdb_5000_movies.csv"))
credits = pd.read_csv(os.path.join(BASE_DIR, "tmdb_5000_credits.csv"))

movies = movies.merge(credits, on='title')
movies = movies[['genres', 'movie_id', 'keywords', 'title', 'cast', 'crew', 'overview']]
movies.dropna(inplace=True)

# ---------- Helper functions for preprocessing ----------
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

def convert_cast(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert_cast)
movies['crew'] = movies['crew'].apply(fetch_director)
movies['overview'] = movies['overview'].apply(lambda x: x.split())

movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])

movies['tags'] = movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew'] + movies['overview']

new_df = movies[['movie_id', 'title', 'tags']].copy()
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

# ---------- Stemming ----------
ps = PorterStemmer()

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_df['tags'] = new_df['tags'].apply(stem)

# ---------- Vectorization (sparse, no full similarity matrix) ----------
# Reduce max_features to save memory on free hosts like Render
cv = CountVectorizer(max_features=3000, stop_words='english')
vectors = cv.fit_transform(new_df['tags'])  # keep sparse matrix, no .toarray()


# ---------- OMDb helper ----------
def fetch_movie_details(movie_title):
    """Fetch poster, rating, year, and plot from OMDb. Uses OMDB_API_KEY env variable."""
    if not OMDB_API_KEY:
        # No key configured: return generic placeholders
        return (
            "https://via.placeholder.com/300x450?text=No+API+Key",
            "N/A",
            "N/A",
            "OMDb API key not configured on server."
        )

    url = f"http://www.omdbapi.com/?t={movie_title}&apikey={OMDB_API_KEY}"
    try:
        data = requests.get(url).json()
        if data.get('Response') == 'True':
            poster = data.get('Poster', "https://via.placeholder.com/300x450?text=No+Image")
            rating = data.get('imdbRating', 'N/A')
            year = data.get('Year', 'N/A')
            plot = data.get('Plot', 'No plot available.')
        else:
            poster = "https://via.placeholder.com/300x450?text=No+Image"
            rating = 'N/A'
            year = 'N/A'
            plot = 'No details available.'
    except Exception:
        poster = "https://via.placeholder.com/300x450?text=Error"
        rating = 'N/A'
        year = 'N/A'
        plot = 'Could not fetch details.'

    return poster, rating, year, plot


# ---------- Recommendation logic (memory-friendly) ----------
def get_recommendations(movie):
    movie = movie.lower().strip()
    titles_lower = new_df['title'].str.lower()

    if movie not in titles_lower.values:
        return []

    movie_index = titles_lower[titles_lower == movie].index[0]

    # Compute similarity ONLY for this movie vs all others (on-demand)
    distances = cosine_similarity(vectors[movie_index], vectors).flatten()

    # Top 5 most similar (skip itself at index 0)
    movies_list = sorted(
        enumerate(distances),
        key=lambda x: x[1],
        reverse=True
    )[1:6]

    recommendations = []
    for idx, score in movies_list:
        title = new_df.iloc[idx].title
        poster, rating, year, plot = fetch_movie_details(title)
        recommendations.append({
            "title": title,
            "poster": poster,
            "rating": rating,
            "year": year,
            "plot": plot
        })

    return recommendations


# ---------- Routes ----------
@app.route("/")
def index():
    movie_list = sorted(new_df['title'].values)
    return render_template("index.html", movies=movie_list)

@app.route("/recommend", methods=["POST"])
def recommend_route():
    data = request.get_json() or {}
    movie = data.get("movie", "")
    recs = get_recommendations(movie)

    if not recs:
        return jsonify({"error": "Movie not found. Try a different title."}), 404

    return jsonify({"recommendations": recs})


# ---------- Local run ----------
if __name__ == "__main__":
    # For local development
    # On Render / production, gunicorn main:app --bind 0.0.0.0:$PORT is used.
    app.run(host="0.0.0.0", port=5000, debug=True)
