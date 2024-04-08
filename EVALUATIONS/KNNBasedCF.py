import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors

class KNNCF:

  def __init__(self, movies, users):
    self.movies = movies
    self.users = users
    self.create_users_knn()
    self.create_movies_knn()

  def normalize(self, ratings_matrix):
    mr = ratings_matrix.copy()
    for iy, ix in np.ndindex(mr.shape):
      if(mr[iy,ix] != 0):
        mr[iy,ix]-=3
    print(mr)
    return mr

  def create_users_knn(self):
    self.users = self.users.join(pd.get_dummies(self.users.gender)+0).drop("gender", axis=1)
    self.users = self.users.join(pd.get_dummies(self.users.occupation)+0).drop("occupation", axis=1)
    self.users = self.users.drop("zip code",  axis=1)
    self.users_X = self.users.drop("user id", axis = 1).to_numpy()
    self.users_Y = self.users["user id"].to_numpy()
    self.knn_users = KNeighborsClassifier(n_neighbors=20)
    self.knn_users.fit(self.users_X, self.users_Y)

  def create_movies_knn(self):
    self.movies = self.movies.drop('movie title', axis = 1).drop('IMDb URL', axis = 1).drop('video release date', axis = 1).drop('release date', axis = 1)
    self.movies_X = self.movies.drop("movie id", axis = 1).to_numpy()
    self.movies_Y = self.movies["movie id"].to_numpy()
    self.knn_movies = KNeighborsClassifier(n_neighbors=20)
    self.knn_movies.fit(self.movies_X, self.movies_Y)

  def recommend_movies(self, ratings_matrix, user_ids, num_recommendations = 10):
    knn = NearestNeighbors(n_neighbors=20)
    knn.fit(ratings_matrix)

    recommendations = []
    for user_id in user_ids:
      user_id = user_id-1
      is_cold_start = np.sum(ratings_matrix[user_id])==0

      weighted_ratings = np.zeros(ratings_matrix.shape[1])
      if(is_cold_start):
        result = self.knn_users.kneighbors([self.users_X[user_id]])
        result = np.column_stack((result[0][0], result[1][0]))
        for [dist, uid] in result:
          weighted_ratings += ratings_matrix[int(uid)]/(1+dist)

      else:
        distances, user_indices = knn.kneighbors(ratings_matrix[user_id].reshape(1, -1), return_distance=True)
        user_indices = user_indices[0][1:]
        distances = distances[0][1:]
        for ii, uid in enumerate(user_indices):
          weighted_ratings+=(ratings_matrix[uid]/(1+distances[ii]))

      recommended_item_indices = np.argsort(weighted_ratings)[::-1]
      unrated_items = [i for i in recommended_item_indices if ratings_matrix[user_id, i] == 0]
      recommendations.append(np.array(unrated_items[:num_recommendations])+1)

    return recommendations

  def recommend_users(self, ratings_matrix, movie_ids, num_recommendations = 10):
    knn = NearestNeighbors(n_neighbors=20)
    knn.fit(ratings_matrix.T)
    recommendations = []
    for movie_id in movie_ids:
      movie_id = movie_id-1
      is_cold_start = np.sum(ratings_matrix[:, movie_id])==0

      weighted_ratings = np.zeros(ratings_matrix.shape[0])
      if(is_cold_start):
        result = self.knn_movies.kneighbors([self.movies_X[movie_id]])
        result = np.column_stack((result[0][0], result[1][0]))
        for [dist, mid] in result:
          dist = 1/(1+dist)
          weighted_ratings += ratings_matrix[:, int(mid)]*dist

      else:
        distances, movie_indices = knn.kneighbors(ratings_matrix[:,movie_id].reshape(1, -1), return_distance=True)
        movie_indices = movie_indices[0][1:]
        distances = distances[0][1:]
        for ii, mid in enumerate(movie_indices):
          weighted_ratings+=(ratings_matrix[:,mid]/(1+distances[ii]))

      recommended_user_indices = np.argsort(weighted_ratings)[::-1]
      unrated_users = [i for i in recommended_user_indices if ratings_matrix[i, movie_id] == 0]
      recommendations.append(np.array(unrated_users[:num_recommendations])+1)

    return recommendations

  def predict_rating(self, ratings_matrix, rating_ids):
    user_ratings_knn = NearestNeighbors(n_neighbors=50)
    movie_ratings_knn = NearestNeighbors(n_neighbors=50)
    user_ratings_knn.fit(ratings_matrix)
    movie_ratings_knn.fit(ratings_matrix.T)
    result = []

    for user_id, movie_id in rating_ids:
      user_id = user_id-1
      movie_id = movie_id-1

      #user based cf
      distances, user_indices = user_ratings_knn.kneighbors(ratings_matrix[user_id].reshape(1, -1), return_distance=True)
      similar_users_indices = user_indices[0][1:]
      similar_users_ratings = []
      for ii, similar_user_index in enumerate(similar_users_indices):
        if(ratings_matrix[similar_user_index, movie_id] != 0):
          similar_users_ratings.append(ratings_matrix[similar_user_index, movie_id])
          
      n1 = np.sum(similar_users_ratings)
      d1 = len(similar_users_ratings)

      #item based cf
      distances, movie_indices = movie_ratings_knn.kneighbors(ratings_matrix.T[movie_id].reshape(1, -1), return_distance=True)
      similar_movie_indices = movie_indices[0][1:]
      similar_movie_ratings = []
      for ii, similar_movie_index in enumerate(similar_movie_indices):
        if(ratings_matrix[user_id, similar_movie_index] != 0):
          similar_movie_ratings.append(ratings_matrix[user_id, similar_movie_index])

      n2 = np.sum(similar_movie_ratings)
      d2 = len(similar_movie_ratings)

      if(d2==0 and d1==0):
        predicted_rating = 3
      elif(d2!=0 and d1!=0):
        predicted_rating = ((n1/d1)+(n2/d2))/2
      elif(d2==0):
        predicted_rating = n1/d1
      else:
        predicted_rating = n2/d2
      result.append(predicted_rating)

    return np.array(result)