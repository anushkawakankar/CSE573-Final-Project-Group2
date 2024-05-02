import numpy as np
import pandas as pd
from SimilarityBasedCF import SimilarityCF

movies_path = "DATA/movies.csv"
users_path = "DATA/users.csv"
ratings_path = "DATA/ratings.csv"

movies = pd.read_csv(movies_path)
users = pd.read_csv(users_path)
ratings = pd.read_csv(ratings_path)

def create_ratings_matrix(ratings):
  matrix = np.zeros((ratings["user id"].max(),ratings["item id"].max()))
  for index, row in ratings.iterrows():
    matrix[row["user id"]-1, row["item id"]-1] = row["rating"]
  return matrix
ratings_matrix = create_ratings_matrix(ratings)

model = SimilarityCF(movies, users, "cosine")
print(list(model.recommend_movies(ratings_matrix,[1])[0]))


   

from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Sample function that could be part of your Python code
def compute_square(number):
    return number * number

@app.route('/square/<int:number>', methods=['GET'])
def get_square(number):
    result = compute_square(number)
    return jsonify({'result': result})

@app.route('/recommend_movies/<int:uid>', methods=['GET'])
def recommend_movies(uid):
    movie_ids = model.recommend_movies(ratings_matrix,[uid])[0]
    # actual_ratings = ratings_matrix[uid-1][movie_ids-1]
    response = jsonify({'movies': list(movies['movie title'].to_numpy()[movie_ids-1]), 
                        # "ratings" : [int(x) for x in actual_ratings], 
                        # "movie_titles": list(movies['movie title'].to_numpy()[movie_ids-1])
                        })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/recommend_users/<int:mid>', methods=['GET'])
def recommend_users(mid):
    user_ids = model.recommend_users(ratings_matrix,[mid])[0]
    # actual_ratings = ratings_matrix[user_ids-1][mid]
    response = jsonify({'users': [int(x) for x in user_ids], 
                        # "ratings" : [int(x) for x in actual_ratings], 
                        # "movie_titles": list(movies['movie title'].to_numpy()[movie_ids-1])
                        })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/predict_rating/<int:uid>/<int:mid>', methods=['GET'])
def predict_rating(uid, mid):
    predicted_rating = model.predict_rating(ratings_matrix,[[uid, mid]])[0]
    actual_rating = ratings_matrix[uid-1][mid-1]
    # actual_ratings = ratings_matrix[user_ids-1][mid]
    response = jsonify({'predicted_rating': predicted_rating, 
                        'actual_rating' : actual_rating
                        # "ratings" : [int(x) for x in actual_ratings], 
                        # "movie_titles": list(movies['movie title'].to_numpy()[movie_ids-1])
                        })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=9999)