import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

class KNNCollaborativeFiltering:
    def __init__(self, movies_df, users_df):
        self.users_df = users_df
        self.movies_df = movies_df
        self.user_knn = None
        self.movie_knn = None
        self.train_knn_models()

    def train_knn_models(self):
        user_features = self.prepare_user_features()
        self.user_knn = NearestNeighbors(metric='cosine')
        self.user_knn.fit(user_features)

        self.movie_knn = NearestNeighbors(metric='cosine')
        movie_features = self.prepare_movie_features()
        self.movie_knn.fit(movie_features)

    def prepare_user_features(self):
        user_features = self.users_df[['age', 'gender', 'occupation']]

        gender_encoder = OneHotEncoder(sparse=False)
        encoded_gender = gender_encoder.fit_transform(user_features[['gender']])
        occupation_encoder = OneHotEncoder(sparse=False)
        encoded_occupation = occupation_encoder.fit_transform(user_features[['occupation']])

        user_features['age'] = user_features['age'] / user_features['age'].max()

        user_features = np.concatenate([user_features[['age']].values, encoded_gender, encoded_occupation], axis=1)

        return user_features

    def prepare_movie_features(self):
        movie_features = self.movies_df.drop(['movie id', 'movie title', 'video release date', 'IMDb URL'], axis=1)

        movie_features['release_year'] = movie_features['release date'].apply(lambda x: int(x[-4:]) if isinstance(x, str) else np.nan)

        movie_features.drop('release date', axis=1, inplace=True)

        imputer = SimpleImputer(strategy='mean')
        movie_features = imputer.fit_transform(movie_features)

        return movie_features

    def recommend_movies(self, ratings_matrix, user_ids, num_recommendations=10):
        recommendations = []

        for user_id in user_ids:
            user_idx = user_id - 1
            rated_movies = np.where(ratings_matrix[user_idx] > 0)[0]

            if len(rated_movies) == 0:
                mean_ratings = np.mean(ratings_matrix, axis=0)
                unrated_top_movies = np.argsort(mean_ratings)[::-1][:num_recommendations]
                recommendations.append([movie_id + 1 for movie_id in unrated_top_movies if movie_id not in rated_movies])
            else:
                distances, user_indices = self.user_knn.kneighbors([self.prepare_user_features()[user_idx]], n_neighbors=20)
                similar_user_indices = user_indices[0][1:]

                predicted_ratings = np.zeros(ratings_matrix.shape[1])

                for sim_user_idx in similar_user_indices:
                    similar_user_ratings = ratings_matrix[sim_user_idx]
                    predicted_ratings += similar_user_ratings

                predicted_ratings /= len(similar_user_indices)

                predicted_ratings[ratings_matrix[user_idx] > 0] = 0

                recommended_movies = np.argsort(predicted_ratings)[::-1][:num_recommendations]

                recommendations.append([movie_id + 1 for movie_id in recommended_movies if movie_id not in rated_movies])

        return recommendations

    def recommend_users(self, ratings_matrix, movie_ids, num_recommendations=10):
        recommendations = []

        for movie_id in movie_ids:
            movie_idx = movie_id - 1
            rated_users = np.where(ratings_matrix[:, movie_idx] > 0)[0]

            if len(rated_users) == 0:
                mean_ratings = np.mean(ratings_matrix, axis=1)
                unrated_top_users = np.argsort(mean_ratings)[::-1][:num_recommendations]
                recommendations.append([user_id + 1 for user_id in unrated_top_users if user_id not in rated_users])
            else:
                distances, movie_indices = self.movie_knn.kneighbors([self.prepare_movie_features()[movie_idx]], n_neighbors=20)
                similar_movie_indices = movie_indices[0][1:]

                predicted_ratings = np.zeros(ratings_matrix.shape[0])

                for sim_movie_idx in similar_movie_indices:
                    similar_movie_ratings = ratings_matrix[:, sim_movie_idx]
                    predicted_ratings += similar_movie_ratings

                predicted_ratings /= len(similar_movie_indices)

                predicted_ratings[ratings_matrix[:, movie_idx] > 0] = 0

                recommended_users = np.argsort(predicted_ratings)[::-1][:num_recommendations]

                recommendations.append([user_id + 1 for user_id in recommended_users if user_id not in rated_users])

        return recommendations

    def calculate_movie_similarity(self, target_movie_idx, similar_movie_indices):
        target_movie_features = self.movies_df.iloc[target_movie_idx, 5:].values
        similar_movie_features = self.movies_df.iloc[similar_movie_indices, 5:].values
        target_movie_features_norm = target_movie_features / np.linalg.norm(target_movie_features)
        similar_movie_features_norm = similar_movie_features / np.linalg.norm(similar_movie_features, axis=1)[:, np.newaxis]
        similarities = np.dot(similar_movie_features_norm, target_movie_features_norm)

        return similarities

    def calculate_user_similarity(self, ratings_matrix, reference_users, target_users):
        reference_ratings = ratings_matrix[reference_users]
        target_ratings = ratings_matrix[target_users]

        similarity_scores = np.dot(reference_ratings, target_ratings.T)

        reference_norms = np.linalg.norm(reference_ratings, axis=1)
        target_norms = np.linalg.norm(target_ratings, axis=1)

        reference_norms = reference_norms[:, np.newaxis]
        target_norms = target_norms[np.newaxis, :]

        similarity_scores /= (reference_norms * target_norms)

        return similarity_scores

    def find_similar_movies(self, movie_idx):
        movie_features = self.prepare_movie_features()
        _, movie_indices = self.movie_knn.kneighbors([movie_features[movie_idx]])
        similar_movie_indices = movie_indices[0][1:]
        return similar_movie_indices

    def find_similar_users(self, user_indices):
        user_features = self.prepare_user_features()
        _, user_indices = self.user_knn.kneighbors(user_features[user_indices])
        similar_user_indices = user_indices[0][1:]
        return similar_user_indices

    def predict_rating(self, ratings_matrix, rating_ids):
        predictions = []

        for user_id, movie_id in rating_ids:
            user_idx = user_id - 1
            movie_idx = movie_id - 1
            _, user_indices = self.user_knn.kneighbors([self.prepare_user_features()[user_idx]])
            similar_user_indices = user_indices[0][1:]
            similar_user_ratings = ratings_matrix[similar_user_indices, movie_idx]
            user_rating_prediction = np.mean(similar_user_ratings) if len(similar_user_ratings) > 0 else np.nan
            _, movie_indices = self.movie_knn.kneighbors([self.prepare_movie_features()[movie_idx]])
            similar_movie_indices = movie_indices[0][1:]
            similar_movie_ratings = ratings_matrix[user_idx, similar_movie_indices]
            movie_rating_prediction = np.mean(similar_movie_ratings) if len(similar_movie_ratings) > 0 else np.nan
            combined_prediction = 5 - (0.5 * user_rating_prediction + 0.5 * movie_rating_prediction)
            predictions.append(combined_prediction)

        return predictions
