import pandas as pd 
from src.recommender import build_similarity_matrix, recommend

movies = pd.read_csv(r"data\raw\ml-latest-small\movies.csv")
ratings = pd.read_csv(r"data\raw\ml-latest-small\ratings.csv")
similarity = build_similarity_matrix(movies)

print("\nGetting recommendations for 'Toy Story (1995)':")
recommendations = recommend("Powder (1995)", movies, similarity)
print(recommendations)