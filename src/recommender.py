import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def build_similarity_matrix(movies):
      tfdif =  TfidfVectorizer(stop_words='english')
      tfdif_matrix = tfdif.fit_transform(movies['genres'])
      similarity  = cosine_similarity(tfdif_matrix)
      return similarity

def recommend(title, movies, similarity, n=6):
    index = movies[movies["title"] == title].index[0]
    scores = list(enumerate(similarity[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_movies = scores[1:n+1]
    return movies.iloc[[i[0] for i in top_movies]]["title"]