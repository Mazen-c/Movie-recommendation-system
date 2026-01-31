#Improting the necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import requests
from dotenv import load_dotenv
import os

# Load variables from .env
load_dotenv()

API_KEY = os.getenv("TMDB_API_KEY")
###########

#REeadinf the dataSet
movies = pd.read_csv(r'data\raw\top10K-TMDB-movies.csv')
###########

#Exploring the data
print("=== DESCRIBE ===")
print(movies.describe())

print("\n=== INFO ===")
print(movies.info())

print("\n=== NULLS ===")
print(movies.isnull().sum())
###########

#Creating a new DataFrame with selected features
movies = movies[['id', 'title', 'genre', 'overview']]

print("\n=== SELECTED FEATURES ===")
print(movies.head())
###########

#Creating a new column called "Tags" which combines genre and overview
movies['tags'] = movies['genre']+movies['overview']
print(movies.head())
###########

#Dropping the columns genre and overview from the new DataFrame
new_data = movies.drop(columns=['genre' , 'overview'])
print("\n=== NEW DATA ===")
print(new_data.head())
###########

#Doing the logic of finding similarity between movies
cv = CountVectorizer(max_features=10000, stop_words='english')
vector = cv.fit_transform(new_data['tags'].values.astype('U')).toarray()
print(vector.shape)
similarity = cosine_similarity(vector)
###########

#Function used to get posters from Api using movie id 
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}"
    data = requests.get(url)
    data = data.json()
    poster_path = data.get('poster_path')
    if poster_path:
        full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
        return full_path
    else:
        return "https://via.placeholder.com/500x750?text=No+Poster"
###########

#Function to recommend movies
def recommend(title):
    index =new_data[new_data['title'] == title].index[0]
    distance= sorted(list(enumerate(similarity[index])) , reverse=True, key=lambda vector: vector[1])
    recommendations = []
    recommendation_posters = []
    for i in distance[1:7]:
        movies_id= movies.iloc[i[0]].id
        recommendations.append(new_data.iloc[i[0]].title)
        recommendation_posters.append(fetch_poster(movies_id))
    return recommendations , recommendation_posters
###########

#Header of the interface 
st.header("Movie Recommendation system")
###########
# Get the list of movie titles
movie_list = movies['title'].values
###########

# Pass the list to the selectbox
selected_movie = st.selectbox("Select your favorite movie", movie_list)
###########

#When the button is pressed, generate recommendations
if st.button("Generate Recommend"):
    movie_re , movie_poster = recommend(selected_movie)
    col1 , col2 , col3 , col4 , col5 , col6= st.columns(6)
    with col1:
        st.text(movie_re[0])
        st.image(movie_poster[0])
    with col2:
        st.text(movie_re[1])
        st.image(movie_poster[1])
    with col3:
        st.text(movie_re[2])
        st.image(movie_poster[2])
    with col4:
        st.text(movie_re[3])
        st.image(movie_poster[3])
    with col5:
        st.text(movie_re[4])
        st.image(movie_poster[4])
    with col6:
     st.text(movie_re[5])
     st.image(movie_poster[5])
###########