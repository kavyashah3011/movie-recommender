import streamlit as st
import pandas as pd
import numpy as np
import ast
import requests
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---- OMDb API Setup ----
OMDB_API_KEY = "28aea51a"  # 🔹 Replace with your OMDb API key

# ---- Load and Prepare Data ----
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")
movies = movies.merge(credits, on='title')
movies = movies[['genres', 'movie_id','keywords','title', 'cast', 'crew', 'overview']]
movies.dropna(inplace=True)

# --- Helper Functions for Data Cleaning ---
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

# Combine all textual features into one column
movies['tags'] = movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew'] + movies['overview']
new_df = movies[['movie_id', 'title', 'tags']].copy()
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

# --- Stemming ---
ps = PorterStemmer()
def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_df['tags'] = new_df['tags'].apply(stem)

# --- Vectorization ---
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
similarity = cosine_similarity(vectors)

# ---- OMDb Poster + Info Fetcher ----
def fetch_movie_details(movie_title):
    """Fetch poster, rating, year, and plot from OMDb"""
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
    except:
        poster = "https://via.placeholder.com/300x450?text=Error"
        rating = 'N/A'
        year = 'N/A'
        plot = 'Could not fetch details.'
    return poster, rating, year, plot

# ---- Recommendation Function ----
def recommend(movie):
    movie = movie.lower()
    if movie not in new_df['title'].str.lower().values:
        return []
    movie_index = new_df[new_df['title'].str.lower() == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommendations = []
    for i in movies_list:
        title = new_df.iloc[i[0]].title
        poster, rating, year, plot = fetch_movie_details(title)
        recommendations.append({
            "title": title,
            "poster": poster,
            "rating": rating,
            "year": year,
            "plot": plot
        })
    return recommendations

# ---- Streamlit UI ----
st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")
st.title("🎥 Movie Recommendation System (with OMDb API)")
st.write("Discover movies similar to your favorite ones!")

movie_list = sorted(new_df['title'].values)
selected_movie = st.selectbox("Select a movie", movie_list)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    if not recommendations:
        st.error("Movie not found. Try a different title.")
    else:
        st.subheader("Top 5 Recommendations:")
        cols = st.columns(5)
        for idx, rec in enumerate(recommendations):
            with cols[idx]:
                st.image(rec["poster"], use_container_width=True)
                st.markdown(f"### {rec['title']}")
                st.markdown(f"**⭐ IMDb:** {rec['rating']}")
                st.markdown(f"**📅 Year:** {rec['year']}")
                st.markdown(f"<small>{rec['plot']}</small>", unsafe_allow_html=True)
