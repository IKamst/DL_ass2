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