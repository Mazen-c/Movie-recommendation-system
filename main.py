import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv(r'C:\Projects\Movie recommendation system\data\raw\top10K-TMDB-movies.csv')

print("=== DESCRIBE ===")
print(movies.describe())

print("\n=== INFO ===")
print(movies.info())

print("\n=== NULLS ===")
print(movies.isnull().sum())

movies = movies[['id', 'title', 'genre', 'overview']]

print("\n=== SELECTED FEATURES ===")
print(movies.head())


movies['tags'] = movies['genre']+movies['overview']
print(movies.head())

new_data = movies.drop(columns=['genre' , 'overview'])
print("\n=== NEW DATA ===")
print(new_data.head())

cv = CountVectorizer(max_features=10000, stop_words='english')
vector = cv.fit_transform(new_data['tags'].values.astype('U')).toarray()
print(vector.shape)
similairty = cosine_similarity(vector)

def recommend(title):
    index =new_data[new_data['title'] == title].index[0]
    distance= sorted(list(enumerate(similairty[index])) , reverse=True, key=lambda vector: vector[1])
    for i in distance[0:5]:
        print(new_data.iloc[i[0]].title)