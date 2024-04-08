import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class SimilarityCF:

  def pearson(self, values):
    C = np.cov(values)
    diag = np.diag(C)
    N = np.sqrt(np.outer(diag, diag))
    N[N == 0] = 1
    return C / N

  def __init__(self, movies, users, similarity_function = "cosine"):
    self.movies = movies
    self.users = users
    if(similarity_function == "pearson"):
      self.similarity_function = self.pearson
    else:
      self.similarity_function = cosine_similarity
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
    self.users_meta_similarity = self.similarity_function(self.users_X)

  def create_movies_knn(self):
    self.movies = self.movies.drop('movie title', axis = 1).drop('IMDb URL', axis = 1).drop('video release date', axis = 1).drop('release date', axis = 1)
    self.movies_X = self.movies.drop("movie id", axis = 1).to_numpy()
    self.movie_meta_similarity = self.similarity_function(self.movies_X)

  def recommend_movies(self, ratings_matrix, user_ids, num_recommendations = 10):
    user_similarity = self.similarity_function(ratings_matrix)
    recommendations = []
    for user_id in user_ids:
      user_id = user_id-1
      is_cold_start = np.sum(ratings_matrix[user_id])==0

      if(is_cold_start):
        similarity_scores = self.users_meta_similarity[user_id]
        weighted_ratings = ratings_matrix.T.dot(similarity_scores)

      else:
        similarity_scores = user_similarity[user_id]
        weighted_ratings = ratings_matrix.T.dot(similarity_scores)

      recommended_item_indices = np.argsort(weighted_ratings)[::-1]
      unrated_items = [i for i in recommended_item_indices if ratings_matrix[user_id, i] == 0]
      recommendations.append(np.array(unrated_items[:num_recommendations])+1)

    return recommendations

  def recommend_users(self, ratings_matrix, movie_ids, num_recommendations = 10):
    movie_similarity = self.similarity_function(ratings_matrix.T)
    recommendations = []
    for movie_id in movie_ids:
      movie_id = movie_id-1
      is_cold_start = np.sum(ratings_matrix[:, movie_id])==0

      if(is_cold_start):
        similarity_scores = self.movie_meta_similarity[movie_id]
        weighted_ratings = ratings_matrix.dot(similarity_scores)
      else:
        similarity_scores = movie_similarity[movie_id]
        weighted_ratings = ratings_matrix.dot(similarity_scores)

      recommended_user_indices = np.argsort(weighted_ratings)[::-1]
      unrated_users = [i for i in recommended_user_indices if ratings_matrix[i, movie_id] == 0]
      recommendations.append(np.array(unrated_users[:num_recommendations])+1)

    return recommendations

  def predict_rating(self, ratings_matrix, rating_ids):
    k=10
    user_similarity = self.similarity_function(ratings_matrix)
    movie_similarity = self.similarity_function(ratings_matrix.T)
    result = []

    for user_id, movie_id in rating_ids:
      user_id = user_id-1
      movie_id = movie_id-1

      #user based cf
      similarity_scores = user_similarity[user_id]
      similar_users_indices = np.argsort(similarity_scores)[::-1][1:k+1]

      new_similar_users_indices = []
      similar_users_ratings = []
      for similar_user_index in similar_users_indices:
        if(ratings_matrix[similar_user_index, movie_id] != 0):
          new_similar_users_indices.append(similar_user_index)
          similar_users_ratings.append(ratings_matrix[similar_user_index, movie_id])

      n1 = np.dot(similarity_scores[new_similar_users_indices], similar_users_ratings)
      d1 = np.sum(np.abs(similarity_scores[new_similar_users_indices]))

      #item based cf
      similarity_scores = movie_similarity[movie_id]
      similar_movie_indices = np.argsort(similarity_scores)[::-1][1:k+1]

      new_similar_movie_indices = []
      similar_movie_ratings = []
      for similar_movie_index in similar_movie_indices:
        if(ratings_matrix[user_id, similar_movie_index] != 0):
          new_similar_movie_indices.append(similar_movie_index)
          similar_movie_ratings.append(ratings_matrix[user_id, similar_movie_index])

      n2 = np.dot(similarity_scores[new_similar_movie_indices], similar_movie_ratings)
      d2 = np.sum(np.abs(similarity_scores[new_similar_movie_indices]))

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