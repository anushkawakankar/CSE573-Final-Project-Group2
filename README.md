# CSE 573 Semantic Web Mining - Final Project

## Movie Recommendation System

This project implements a movie recommendation system on the MovieLens 100K dataset using four different algorithms:

- Collaborative Filtering with K-Nearest Neighbors (KNN)
- Matrix Factorization-Based Collaborative Filtering
- Similarity-Based Collaborative Filtering
- LlamaRec: Two-Stage Recommendation with Large Language Models (LLMs)

We test our algorithms using five different scenarios that play a cruicial role in determining the performance of a movie recommendation system:

1. Recommend movies to a new user (cold start)
2. Recommend users for a new movie (cold start)
3. Recommend movies to an existing user
4. Recommend users for an existing movie
5. Predict ratings that a user might give a movie

## Getting Started

### To run the UI

1. Start the back-end server from the root of this repository using the command:

   `python CODE/main.py`

2. Using a local server (such as HTTP Simple Server), open the **CODE/index.html** file.

### Python dependencies

1. Numpy
2. Pandas
3. Flask
4. Scikit-learn
