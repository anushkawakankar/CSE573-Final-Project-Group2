import numpy as np

max_num_test = 100

def rmse(values_1, values_2):
  return np.sqrt(np.square(values_1 - values_2).mean())

def precision(true_values, predicted_values):
  true_values = set(true_values)
  predicted_values = set(predicted_values)
  intersection = true_values.intersection(predicted_values)
  return len(intersection)/len(predicted_values)

def evaluate_scenario_1(model, ratings_matrix, num_recommendations = 10):
  #New user - cold start
  user_indexes = []
  for ii, ratings in enumerate(ratings_matrix):
    if(len(np.where(ratings>3)[0]) > 2*num_recommendations):
      user_indexes.append(ii)
  user_indexes = np.array(user_indexes[:max_num_test])

  modified_ratings = ratings_matrix.copy()
  modified_ratings[user_indexes] = 0

  recommended_movies = model.recommend_movies(modified_ratings, user_indexes+1, num_recommendations)
  true_values = [np.where(ratings_matrix[user_index]>3)[0]+1 for user_index in user_indexes]
  precisions = np.array([precision(true_values[ii], recommended_movies[ii]) for ii in range(len(user_indexes))])
  return precisions.mean()

def evaluate_scenario_2(model, ratings_matrix, num_recommendations = 10):
  #New movie - cold start
  movie_indexes = []
  for ii, ratings in enumerate(ratings_matrix.T):
    if(len(np.where(ratings>3)[0]) > 2*num_recommendations):
      movie_indexes.append(ii)
  movie_indexes = np.array(movie_indexes[:max_num_test])

  modified_ratings = ratings_matrix.copy()
  modified_ratings[:,movie_indexes] = 0

  recommended_users = model.recommend_users(modified_ratings, movie_indexes+1, num_recommendations)
  true_values = [np.where(ratings_matrix.T[movie_index]>3)[0]+1 for movie_index in movie_indexes]
  precisions = np.array([precision(true_values[ii], recommended_users[ii]) for ii in range(len(movie_indexes))])
  return precisions.mean()

def evaluate_scenario_3(model, ratings_matrix, num_recommendations = 10):
  #Given user
  user_indexes = []
  for ii, ratings in enumerate(ratings_matrix):
    if(len(np.where(ratings>3)[0]) > 4*num_recommendations):
      user_indexes.append(ii)
  user_indexes = np.array(user_indexes[:max_num_test])

  modified_ratings = ratings_matrix.copy()
  true_values = []
  for user_index in user_indexes:
    removed_indexes = np.where(ratings_matrix[user_index]>3)[0][::2]
    true_values.append(removed_indexes+1)
    modified_ratings[user_index, removed_indexes] = 0

  recommended_movies = model.recommend_movies(modified_ratings, user_indexes+1, num_recommendations)
  precisions = np.array([precision(true_values[ii], recommended_movies[ii]) for ii in range(len(user_indexes))])
  return precisions.mean()

def evaluate_scenario_4(model, ratings_matrix, num_recommendations = 10):
  #Given movie
  movie_indexes = []
  for ii, ratings in enumerate(ratings_matrix.T):
    if(len(np.where(ratings>3)[0]) > 4*num_recommendations):
      movie_indexes.append(ii)
  movie_indexes = np.array(movie_indexes[:max_num_test])

  modified_ratings = ratings_matrix.copy()
  true_values = []
  for movie_index in movie_indexes:
    removed_indexes = np.where(ratings_matrix.T[movie_index]>3)[0][::2]
    true_values.append(removed_indexes+1)
    modified_ratings[removed_indexes, movie_index] = 0

  recommended_users = model.recommend_users(modified_ratings, movie_indexes+1, num_recommendations)
  precisions = np.array([precision(true_values[ii], recommended_users[ii]) for ii in range(len(movie_indexes))])
  return precisions.mean()

def evaluate_scenario_5(model, ratings_matrix, num_recommendations = 10):
  #Predict rating
  user_ids = np.array(range(0,ratings_matrix.shape[0],20))+1
  movie_ids = np.array(range(0,ratings_matrix.shape[1],20))+1

  rating_ids = []
  for uid in user_ids:
    for mid in movie_ids:
      if(ratings_matrix[uid-1, mid-1]!=0):
        rating_ids.append((uid, mid))

  true_values = np.array([ratings_matrix[uid-1, mid-1] for uid, mid in rating_ids])
  predicted_values = model.predict_rating(ratings_matrix, rating_ids)
  return rmse(true_values, predicted_values)

