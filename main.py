from fastapi import FastAPI, Query
import pandas as pd
import numpy as np
import joblib
from typing import List
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import sklearn
print('The scikit-learn version is {}.'.format(sklearn.__version__))
app = FastAPI()
origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Load the preprocessed data, anime_ids, and model pipeline
X = joblib.load('X_preprocessed.pkl')
model_pipeline = joblib.load('anime_recommendation_model.pkl')

data = pd.read_csv('anime_filtered.csv')

genre_dummies = data['genre'].str.get_dummies(sep=', ')
data = pd.concat([data, genre_dummies], axis=1)

# Drop the original 'genre' column
data = data.drop(columns=['genre'])
# Function to create a user profile based on favorited animes
def create_user_profile(favorited_anime_ids, original_data, X_preprocessed):
    favorited_indices = original_data.index[original_data['anime_id'].isin(favorited_anime_ids)].tolist()
    favorited_features = X_preprocessed[favorited_indices]
    user_profile = np.asarray(favorited_features.mean(axis=0))
    return user_profile

# Function to get recommendations for a user
def get_recommendations_for_user(favorited_anime_ids, n_recommendations=5):
    user_profile = create_user_profile(favorited_anime_ids, data, X)
    user_profile = user_profile.reshape(1, -1)
    distances, indices = model_pipeline.named_steps['nearestneighbors'].kneighbors(user_profile, n_neighbors=n_recommendations + len(favorited_anime_ids))
    recommendations = data.iloc[indices[0]].copy()
    recommendations['distance'] = distances[0]
    recommendations = recommendations[~recommendations['anime_id'].isin(favorited_anime_ids)]  # Exclude already favorited animes
    return recommendations[[ 'title']].head(n_recommendations)



@app.get("/")
def read_root():
    return {"message": "Welcome to my FastAPI app!"}


@app.get("/search")
def search_titles(query: str = Query(..., description="The string to search for in titles")):
    query_lower = query.lower()
    
    if data.empty:
        print('Data is empty')
    
    matching_titles = data[data['title'].str.lower().str.startswith(query_lower, na=False)][['anime_id', 'title']].to_dict(orient='records')
    
    return {"matching_titles": matching_titles}


class ArrayModel(BaseModel):
    arr: List[int]

@app.post("/recommendations")
def recommendations(array_model: ArrayModel ):
    ids = array_model.arr
    recommendations = get_recommendations_for_user(ids, n_recommendations=10)
    return recommendations.to_dict(orient='records')