from Task_2.data_exploration import *
from Task_2.preprocessing import *
from Task_2.ncf import *

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
#print(train_users)

num_users = train_users.nunique()
num_items = train_movies.nunique()

# Links: https://sparsh-ai.github.io/rec-tutorials/matrixfactorization%20movielens%20pytorch%20scratch/2021/04/21/rec-algo-ncf-pytorch-pyy0715.html

# Links: https://caravanuden.com/spotify_recsys/neural_collaborative_filtering.html