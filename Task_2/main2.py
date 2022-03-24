from Task_2.data_exploration import *
from Task_2.preprocessing import *
from Task_2.ncf import *


# Links: https://sparsh-ai.github.io/rec-tutorials/matrixfactorization%20movielens%20pytorch%20scratch/2021/04/21/rec-algo-ncf-pytorch-pyy0715.html

# Links: https://caravanuden.com/spotify_recsys/neural_collaborative_filtering.html

# Links: https://colab.research.google.com/github/nipunbatra/blog/blob/master/_notebooks/2017-12-29-neural-collaborative-filtering.ipynb

 # Of vanaf library: https://towardsdatascience.com/a-complete-guide-to-recommender-system-tutorial-with-sklearn-surprise-keras-recommender-5e52e8ceace1
# https://github.com/microsoft/recommenders/blob/main/examples/02_model_hybrid/ncf_deep_dive.ipynb


# TODO:
# Build parse one-hot encoding matrix that maps relationships between users (playlists) and items (tracks).
# Playlist (more generally called u for user) and item (i) vectors are used to create embeddings (low-dimensional) for each playlist and item.
# Generalized Matrix Factorization (GMF) combines the two embeddings using the dot product (this is the classic matrix factorization).
# Multi-layer perceptron (MLP) can also create embeddings for user and items. However, instead of taking a dot product of these to obtain the rating, I can concatenate them to create a feature vector that is passed on to deeper layers.
# NeuMF then combines the predictions from MLP and GMF to obtain the final prediction

#%%
print('--------- load data ----------')
ratings = load_data()
ratings_small = create_small_practice_dataset(ratings)

#%%
# print('---------- data exploration -------------')
# data_exploration(ratings_small)

#%%
print('---------- pre-processing -----------')
train, test = train_test_split(ratings_small)
train = transform_to_implicit(train)
train_users, train_movies, train_labels = add_negatives(train, ratings_small)
train_data = pd.DataFrame(list(zip(train_users, train_movies, train_labels)),
               columns =['userID', 'movieID', 'Label'])

print(train_data)

data = reindex_ID(train_data) # TODO: do same transformation for test data

# print(data)



